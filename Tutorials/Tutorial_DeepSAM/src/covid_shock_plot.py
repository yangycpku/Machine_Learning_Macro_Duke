import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import numpy as np
import torch
import torch.optim as optim
import scipy.io
from numpy import linalg as LA
from pathlib import Path
import importlib
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
from getpass import getpass
import os, pathlib
from omegaconf import OmegaConf




# ---------------------------------
# helper: compute U, V, P, and F
# ---------------------------------
@torch.no_grad()
def compute_UVPF(ct, pinn_S, z_scalar, g_flat):
    z_batch = torch.tensor([[float(z_scalar)]], device=ct.device, dtype=torch.float32)
    S_grid, alphas = ct.calculate_alphas(pinn_S, z_batch, g_flat)
    # g_e, g_u, U, g_p, g_v, V, g_f = ct.marginals_U_V(g_flat, S_grid, alphas)
    g_e, g_u, U, g_p, g_v, V, g_f = ct.marginals_U_V(g_flat, S_grid, alphas, z_batch)
    P = g_flat.reshape(1, ct.nx, ct.ny).mean(dim=(1,2), keepdim=True)
    F = P + V
    return float(U.item()), float(V.item()), float(P.item()), float(F.item())

# ---------------------------------
# helper: tensor/array -> numpy
# ---------------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# ---------------------------------
# one single LH path after COVID
# start_state must be "L" or "H"
# ---------------------------------
def generate_single_LH_path_from_start(ct, start_state, T_steps, dt, seed=None, fixed_z=False):
    if start_state not in ["L", "H"]:
        raise ValueError("start_state must be 'L' or 'H'")

    lam_LH = float(ct.lam_LH)
    lam_HL = float(ct.lam_HL)

    p_LH = 1.0 - np.exp(-lam_LH * dt)
    p_HL = 1.0 - np.exp(-lam_HL * dt)

    rng = np.random.default_rng(seed)

    s = 0 if start_state == "L" else 1
    z_np = np.empty((1, T_steps), dtype=np.float32)
    z_np[0, 0] = float(s)

    for t in range(1, T_steps):
        u = rng.random()
        if not fixed_z:
            if s == 0:
                if u < p_LH:
                    s = 1
            else:
                if u < p_HL:
                    s = 0
        z_np[0, t] = float(s)

    return torch.tensor(z_np, device=ct.device, dtype=torch.float32)

# ---------------------------------
# Figure 3.a blue line, one path only
# ---------------------------------
@torch.no_grad()
def simulate_figure3a_blue_one_path(
    ct,
    pinn_S,
    gm_low,
    gm_high,
    g0_bar,
    z_pre_scalar,
    z_after_start="L",
    z_rel="bad", # or bad
    dt=1e-2,
    T_end=1.2,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
    seed=123,
    k_mu=2,
    fixed_z=False,
):
    """
    One-path version:
    pre-COVID ergodic g0_bar -> D for t_covid -> one LH path
    """

    g0_flat = torch.as_tensor(g0_bar, device=ct.device, dtype=torch.float32).reshape(1, -1)

    T = int(round(T_end / dt))
    H = int(round(t_covid / dt))

    if H < 1:
        raise ValueError("t_covid is too small relative to dt")
    if H >= T:
        raise ValueError("t_covid must be strictly smaller than T_end")

    # D = 2
    z_D = 2.0

    # one single LH path after COVID
    T_post = T - H
    z_post = generate_single_LH_path_from_start(
        ct=ct,
        start_state=z_after_start,
        T_steps=T_post,
        dt=dt,
        seed=seed,
        fixed_z=fixed_z,
    )

    # full path: D first, then LH
    z_full = torch.full((1, T), z_D, device=ct.device, dtype=torch.float32)
    z_full[:, H:] = z_post

    # pre-COVID employment level for normalization
    if z_rel == "good":
        P0 = np.mean(gm_low)
    elif z_rel == "bad":
        P0 = np.mean(gm_high)
    else:
        raise ValueError("z_rel must be 'good' or 'bad'")
    # P0 = float(g0_flat.reshape(1, ct.nx, ct.ny).mean().item())

    out = simulate_path(
        ct=ct,
        pinn_S=pinn_S,
        z_path=z_full,
        g_init=g0_flat.clone(),
        substeps=substeps,
        z_pre=float(z_pre_scalar),
        clamp_g=clamp_g,
        k_mu=k_mu,
    )

    P_path = np.asarray(out["P"], dtype=float)
    rel_emp = 100.0 * (P_path / P0)

    return {
        "t": np.asarray(out["t"], dtype=float),
        "rel_emp": rel_emp,
        "P": P_path,
        "P0": P0, # This rather relative P.
        "z_path": np.asarray(out["z"], dtype=float),
        "out_raw": out,
    }



@torch.no_grad()
def simulate_path(ct, pinn_S, z_path, g_init, substeps, z_pre=None, clamp_g=True, k_mu=2, dt=0.01):
    """
    Simulate one deterministic z-path with:
    - drift update over each dt using current z
    - jump in g at every z-switch: g <- (1 - sigma(z,z')) * g
    - optional initial boundary jump at t=0 from z_pre -> z_path[0]
    - muF/F computed on a STRICT forward-looking window [k, k+k_mu]
    - muF/F = NaN if:
        * current point is a jump point
        * any jump occurs in (k, k+k_mu]
        * there are not enough future points for a full k_mu window
    """
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer. Got: {substeps}")
    if not isinstance(k_mu, int) or k_mu < 1:
        raise ValueError(f"k_mu must be a positive integer. Got: {k_mu}")

    g = g_init.clone()
    sigma_mat_np = ct.sigma_mat.detach().cpu().numpy()
    T_local = z_path.shape[1]

    # state at t=0+
    z0 = float(z_path[0, 0].item())

    # if not provided, default: no initial boundary jump
    if z_pre is None:
        z_pre = z0
    else:
        z_pre = float(z_pre)

    # ---------- initial boundary jump at t=0 ----------
    initial_jump = (z_pre != z0)
    if initial_jump:
        # pre-jump aggregate at t=0- under z_pre
        _, _, _, F_pre0 = compute_UVPF(ct, pinn_S, z_pre, g)

        z_from0 = int(round(z_pre))
        z_to0   = int(round(z0))
        sigma0 = float(sigma_mat_np[z_from0, z_to0])

        # apply jump exactly at t=0
        g = (1.0 - sigma0) * g
        if clamp_g:
            g = g.clamp(min=0.0)

        # post-jump aggregate at t=0+ under z0
        _, _, _, F_post0 = compute_UVPF(ct, pinn_S, z0, g)
        sigmaF0 = (F_post0 - F_pre0) / F_pre0
    else:
        sigmaF0 = np.nan

    # record t=0 from post-jump state (if any)
    U0, V0, P0, F0 = compute_UVPF(ct, pinn_S, z0, g)

    t_list = [0.0]
    z_list = [z0]
    U_list = [U0]
    V_list = [V0]
    P_list = [P0]
    F_list = [F0]

    sigmaF_over_F_list = [sigmaF0]
    jump_flags = [initial_jump]

    # ---------- main loop over t_k -> t_{k+1} ----------
    for k in range(T_local - 1):
        z_k = float(z_path[0, k].item())
        z_batch = torch.tensor([[z_k]], device=ct.device, dtype=torch.float32)

        # (A) Drift over [t_k, t_{k+1}) using z_k
        for _ in range(substeps):
            S_grid, alphas = ct.calculate_alphas(pinn_S, z_batch, g)
            g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_grid, alphas, z_batch)

            U_safe = torch.clamp(U_t, min=1e-12)
            V_safe = torch.clamp(V_t, min=1e-12)
            M_u = ct.m(U_safe, V_safe) / U_safe

            mu = ct.mu_g(g, M_u, V_safe, alphas, g_e, g_p, g_f, z_batch)
            g = g + (dt / substeps) * mu
            if clamp_g:
                g = g.clamp(min=0.0)

        # (B) Jump at t_{k+1} if z switches
        z_next = float(z_path[0, k + 1].item())
        is_jump = (z_next != z_k)
        jump_flags.append(is_jump)

        if is_jump:
            # pre-jump F at t_{k+1}- under z_k
            _, _, _, F_pre = compute_UVPF(ct, pinn_S, z_k, g)

            z_from = int(round(z_k))
            z_to   = int(round(z_next))
            sigma_jump = float(sigma_mat_np[z_from, z_to])

            # apply jump exactly at switch
            g = (1.0 - sigma_jump) * g
            if clamp_g:
                g = g.clamp(min=0.0)

            # post-jump F at t_{k+1}+ under z_next
            _, _, _, F_post = compute_UVPF(ct, pinn_S, z_next, g)
            sigmaF_over_F_list.append((F_post - F_pre) / F_pre)
        else:
            sigmaF_over_F_list.append(np.nan)

        # store variables at t_{k+1} from post-jump state
        U1, V1, P1, F1 = compute_UVPF(ct, pinn_S, z_next, g)

        t_list.append((k + 1) * dt)
        z_list.append(z_next)
        U_list.append(U1)
        V_list.append(V1)
        P_list.append(P1)
        F_list.append(F1)

    F_arr = np.array(F_list, dtype=float)
    jump_flags_arr = np.array(jump_flags, dtype=bool)
    n = len(F_arr)

    # STRICT forward-looking muF/F over full k_mu * dt window
    mu_over_F = np.full_like(F_arr, np.nan)

    for k in range(n):
        # need full forward window
        if k + k_mu >= n:
            continue

        # current point itself is a jump point
        if jump_flags_arr[k]:
            continue

        # any jump inside the forward window contaminates the window
        if np.any(jump_flags_arr[k + 1 : k + k_mu + 1]):
            continue

        mu_over_F[k] = (F_arr[k + k_mu] - F_arr[k]) / (k_mu * dt * F_arr[k])

    return {
        "t": np.array(t_list),
        "z": np.array(z_list),
        "U": np.array(U_list),
        "V": np.array(V_list),
        "P": np.array(P_list),
        "F": F_arr,
        "muF_over_F": mu_over_F,
        "sigmaF_over_F": np.array(sigmaF_over_F_list, dtype=float),
        "jump_flags": jump_flags_arr,
    }




# # ---------------------------------
# # helper: tensor/array -> numpy
# # ---------------------------------
# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)


# ---------------------------------
# restricted one-path simulation:
# alpha is always evaluated at g_erg
# but g_t still evolves over time
# ---------------------------------
@torch.no_grad()
def simulate_path_restricted_one_path(
    ct,
    pinn_S,
    z_path,
    g_init,
    g_erg,
    substeps,
    z_pre=None,
    clamp_g=True,
    dt=0.01,
):
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer. Got: {substeps}")

    # z_path -> tensor shape (1, T)
    if not torch.is_tensor(z_path):
        z_path_t = torch.as_tensor(z_path, device=ct.device, dtype=torch.float32)
    else:
        z_path_t = z_path.to(device=ct.device, dtype=torch.float32)

    if z_path_t.ndim == 1:
        z_path_t = z_path_t.unsqueeze(0)

    if z_path_t.shape[0] != 1:
        raise ValueError(f"z_path must be one path only. Got shape {tuple(z_path_t.shape)}")

    g = torch.as_tensor(g_init, device=ct.device, dtype=torch.float32).reshape(1, -1).clone()
    g_erg_t = torch.as_tensor(g_erg, device=ct.device, dtype=torch.float32).reshape(1, -1)

    sigma_mat_np = to_numpy(ct.sigma_mat)
    T_local = z_path_t.shape[1]

    # precompute S_erg[z], alpha_erg[z] at fixed ergodic distribution
    S_list = []
    alpha_list = []
    for z_val in [0, 1, 2]:
        z_batch = torch.tensor([[float(z_val)]], device=ct.device, dtype=torch.float32)
        S_z, alpha_z = ct.calculate_alphas(pinn_S, z_batch, g_erg_t)
        S_list.append(S_z[0])         # (nx, ny)
        alpha_list.append(alpha_z[0]) # (nx, ny)

    S_erg_stack = torch.stack(S_list, dim=0)           # (3, nx, ny)
    alpha_erg_stack = torch.stack(alpha_list, dim=0)   # (3, nx, ny)

    def compute_UVPF_restricted(z_scalar, g_flat):
        z_idx = int(round(float(z_scalar)))
        z_batch = torch.tensor([[float(z_scalar)]], device=ct.device, dtype=torch.float32)
        S_now = S_erg_stack[z_idx].unsqueeze(0)
        alpha_now = alpha_erg_stack[z_idx].unsqueeze(0)
        g_e, g_u, U, g_p, g_v, V, g_f = ct.marginals_U_V(g_flat, S_now, alpha_now, z_batch)
        P = g_flat.reshape(1, ct.nx, ct.ny).mean(dim=(1, 2), keepdim=True)
        F = P + V
        return float(U.item()), float(V.item()), float(P.item()), float(F.item())

    # state at t=0+
    z0 = float(z_path_t[0, 0].item())

    if z_pre is None:
        z_pre = z0
    else:
        z_pre = float(z_pre)

    # ---------- initial boundary jump at t=0 ----------
    initial_jump = (z_pre != z0)
    if initial_jump:
        _, _, _, F_pre0 = compute_UVPF_restricted(z_pre, g)

        z_from0 = int(round(z_pre))
        z_to0   = int(round(z0))
        sigma0 = float(sigma_mat_np[z_from0, z_to0])

        g = (1.0 - sigma0) * g
        if clamp_g:
            g = g.clamp(min=0.0)

        _, _, _, F_post0 = compute_UVPF_restricted(z0, g)
        sigmaF0 = (F_post0 - F_pre0) / F_pre0
    else:
        sigmaF0 = np.nan

    # record t=0 from post-jump state
    U0, V0, P0, F0 = compute_UVPF_restricted(z0, g)

    t_list = [0.0]
    z_list = [z0]
    U_list = [U0]
    V_list = [V0]
    P_list = [P0]
    F_list = [F0]

    sigmaF_over_F_list = [sigmaF0]
    jump_flags = [initial_jump]

    # ---------- main loop ----------
    for k in range(T_local - 1):
        z_k = float(z_path_t[0, k].item())
        z_idx = int(round(z_k))
        z_batch = torch.tensor([[z_k]], device=ct.device, dtype=torch.float32)

        # fixed S and alpha evaluated at g_erg, but current z_k
        S_t = S_erg_stack[z_idx].unsqueeze(0)
        alpha_t = alpha_erg_stack[z_idx].unsqueeze(0)

        for _ in range(substeps):
            g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_t, alpha_t, z_batch)

            U_safe = torch.clamp(U_t, min=1e-12)
            V_safe = torch.clamp(V_t, min=1e-12)
            M_u = ct.m(U_safe, V_safe) / U_safe

            mu = ct.mu_g(g, M_u, V_safe, alpha_t, g_e, g_p, g_f, z_batch)
            g = g + (dt / substeps) * mu
            if clamp_g:
                g = g.clamp(min=0.0)

        # jump at t_{k+1}
        z_next = float(z_path_t[0, k + 1].item())
        is_jump = (z_next != z_k)
        jump_flags.append(is_jump)

        if is_jump:
            _, _, _, F_pre = compute_UVPF_restricted(z_k, g)

            z_from = int(round(z_k))
            z_to   = int(round(z_next))
            sigma_jump = float(sigma_mat_np[z_from, z_to])

            g = (1.0 - sigma_jump) * g
            if clamp_g:
                g = g.clamp(min=0.0)

            _, _, _, F_post = compute_UVPF_restricted(z_next, g)
            sigmaF_over_F_list.append((F_post - F_pre) / F_pre)
        else:
            sigmaF_over_F_list.append(np.nan)

        U1, V1, P1, F1 = compute_UVPF_restricted(z_next, g)

        t_list.append((k + 1) * dt)
        z_list.append(z_next)
        U_list.append(U1)
        V_list.append(V1)
        P_list.append(P1)
        F_list.append(F1)

    return {
        "t": np.array(t_list),
        "z": np.array(z_list),
        "U": np.array(U_list),
        "V": np.array(V_list),
        "P": np.array(P_list),
        "F": np.array(F_list),
        "sigmaF_over_F": np.array(sigmaF_over_F_list, dtype=float),
        "jump_flags": np.array(jump_flags, dtype=bool),
    }


# ---------------------------------
# wrapper for Figure 3.a orange line
# uses the SAME z-path as the blue line
# ---------------------------------
@torch.no_grad()
def simulate_figure3a_orange_one_path(
    ct,
    pinn_S,
    gm_low,
    gm_high,
    g0_bar,
    z_pre_scalar,
    z_path_full,
    substeps=1,
    clamp_g=True,
    z_rel="bad",
):
    g0_flat = torch.as_tensor(g0_bar, device=ct.device, dtype=torch.float32).reshape(1, -1)
    z_path_t = torch.as_tensor(z_path_full, device=ct.device, dtype=torch.float32)

    if z_path_t.ndim == 1:
        z_path_t = z_path_t.unsqueeze(0)

    out = simulate_path_restricted_one_path(
        ct=ct,
        pinn_S=pinn_S,
        z_path=z_path_t,
        g_init=g0_flat.clone(),
        g_erg=g0_flat.clone(),
        substeps=substeps,
        z_pre=float(z_pre_scalar),
        clamp_g=clamp_g,
    )

    # P_pre = float(g0_flat.reshape(1, ct.nx, ct.ny).mean().item())
    if z_rel == "good":
        P_pre = np.mean(gm_low)
    elif z_rel == "bad":
        P_pre = np.mean(gm_high)
    else:
        raise ValueError("z_rel must be 'good' or 'bad'")

    rel_emp = 100.0 * (np.asarray(out["P"], dtype=float) / P_pre)

    return {
        "t": np.asarray(out["t"], dtype=float),
        "rel_emp": rel_emp,
        "P": np.asarray(out["P"], dtype=float),
        "P_pre": P_pre, # This is rather relative P.
        "z_path": np.asarray(out["z"], dtype=float),
        "out_raw": out,
    }



# -----------------------------
# helper: sigma reference line for return jump: z_shock -> z_base
# -----------------------------
def local_sigma_ref_series(out, sigma_mat, z_pre=None, window_steps=2):
    """
    Full-length (same length as out['t']) reference series:
    NaN everywhere except small windows around jump times,
    where it is -sigma(z_from, z_to).
    """
    t = np.asarray(out["t"], dtype=float)
    z = np.asarray(out["z"], dtype=float)
    jump_flags = np.asarray(out["jump_flags"], dtype=bool)

    ref = np.full(t.shape, np.nan, dtype=float)
    n = len(t)

    for k in range(n):
        if not jump_flags[k]:
            continue

        # Jump mapping
        if k == 0:
            # initial boundary jump at t=0: z_pre -> z[0]
            if z_pre is None:
                continue
            z_from = int(round(float(z_pre)))
            z_to   = int(round(float(z[0])))
        else:
            # regular jump at t_k: z[k-1] -> z[k]
            z_from = int(round(float(z[k - 1])))
            z_to   = int(round(float(z[k])))

        val = -float(sigma_mat[z_from, z_to])

        lo = max(0, k - int(window_steps))
        hi = min(n - 1, k + int(window_steps))
        ref[lo:hi + 1] = val

    return ref



####################################################################
####################### many simulation block
####################################################################


# --------------------------------------------------
# Generic COVID-block simulator
# mode = "full" or "restricted"
# --------------------------------------------------
@torch.no_grad()
def simulate_covid_block(
    ct,
    pinn_S,
    g0_bar,
    z_pre_scalar,
    mode="full",
    g_erg=None,
    dt=1e-2,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
):
    """
    Simulate the COVID block 0 <= t <= t_covid with:
      - initial jump at t=0 from z_pre_scalar in {L,H} to D
      - drift in D afterward
      - ergodic normalization from g0_bar

    Parameters
    ----------
    g0_bar : array-like
        Pre-COVID ergodic distribution.
    z_pre_scalar : float
        Pre-COVID aggregate state in {0,1}.
    mode : str
        "full" or "restricted".
    g_erg : array-like or None
        Ergodic reference distribution for restricted dynamics.
        If None and mode=="restricted", defaults to g0_bar.
    dt : float
        Time step.
    t_covid : float
        Length of the COVID block.
    substeps : int
        Number of drift substeps per dt.
    clamp_g : bool
        Whether to clamp g to be nonnegative.

    Returns
    -------
    dict with:
      t       : times from 0 to t_covid inclusive
      z_path  : all-D path on that interval
      U, V, P, F
      rel_emp : 100 * P / P0
      P0      : ergodic pre-COVID employment reference
      g_end   : g at t = t_covid, still in D
      z_end   : equals 2.0
      mode    : "full" or "restricted"
    """
    if mode not in ["full", "restricted"]:
        raise ValueError("mode must be 'full' or 'restricted'")
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer. Got: {substeps}")

    H = int(round(t_covid / dt))
    if H < 1:
        raise ValueError("t_covid is too small relative to dt.")

    z_D = 2.0
    z_pre = float(z_pre_scalar)

    g0_flat = torch.as_tensor(g0_bar, device=ct.device, dtype=torch.float32).reshape(1, -1)
    g = g0_flat.clone()

    # old ergodic normalization
    P0 = float(g0_flat.reshape(1, ct.nx, ct.ny).mean().item())

    sigma_mat_np = ct.sigma_mat.detach().cpu().numpy()

    # --------------------------------------------------
    # restricted objects, if needed
    # --------------------------------------------------
    if mode == "restricted":
        if g_erg is None:
            g_erg = g0_bar

        g_erg_t = torch.as_tensor(g_erg, device=ct.device, dtype=torch.float32).reshape(1, -1)

        S_list = []
        alpha_list = []
        for z_val in [0, 1, 2]:
            z_batch = torch.tensor([[float(z_val)]], device=ct.device, dtype=torch.float32)
            S_z, alpha_z = ct.calculate_alphas(pinn_S, z_batch, g_erg_t)
            S_list.append(S_z[0])         # (nx, ny)
            alpha_list.append(alpha_z[0]) # (nx, ny)

        S_erg_stack = torch.stack(S_list, dim=0)           # (3, nx, ny)
        alpha_erg_stack = torch.stack(alpha_list, dim=0)   # (3, nx, ny)

        def compute_UVPF_restricted(z_scalar, g_flat):
            z_idx = int(round(float(z_scalar)))
            z_batch = torch.tensor([[float(z_scalar)]], device=ct.device, dtype=torch.float32)

            S_now = S_erg_stack[z_idx].unsqueeze(0)
            alpha_now = alpha_erg_stack[z_idx].unsqueeze(0)

            g_e, g_u, U, g_p, g_v, V, g_f = ct.marginals_U_V(g_flat, S_now, alpha_now, z_batch)
            P = g_flat.reshape(1, ct.nx, ct.ny).mean(dim=(1, 2), keepdim=True)
            F = P + V

            return float(U.item()), float(V.item()), float(P.item()), float(F.item())

    # --------------------------------------------------
    # initial jump at t=0: z_pre -> D
    # --------------------------------------------------
    if z_pre != z_D:
        z_from0 = int(round(z_pre))
        z_to0 = int(round(z_D))
        sigma0 = float(sigma_mat_np[z_from0, z_to0])

        g = (1.0 - sigma0) * g
        if clamp_g:
            g = g.clamp(min=0.0)

    # --------------------------------------------------
    # record post-jump state at t=0
    # --------------------------------------------------
    if mode == "full":
        U0, V0, P_init, F0 = compute_UVPF(ct, pinn_S, z_D, g)
    else:
        U0, V0, P_init, F0 = compute_UVPF_restricted(z_D, g)

    t_list = [0.0]
    z_list = [z_D]
    U_list = [U0]
    V_list = [V0]
    P_list = [P_init]
    F_list = [F0]

    # --------------------------------------------------
    # deterministic drift in D over [0, t_covid]
    # --------------------------------------------------
    zD_batch = torch.tensor([[z_D]], device=ct.device, dtype=torch.float32)

    if mode == "restricted":
        S_D = S_erg_stack[2].unsqueeze(0)
        alpha_D = alpha_erg_stack[2].unsqueeze(0)

    for k in range(H):
        for _ in range(substeps):
            if mode == "full":
                S_grid, alphas = ct.calculate_alphas(pinn_S, zD_batch, g)
            else:
                S_grid = S_D
                alphas = alpha_D

            g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_grid, alphas, zD_batch)

            U_safe = torch.clamp(U_t, min=1e-12)
            V_safe = torch.clamp(V_t, min=1e-12)
            M_u = ct.m(U_safe, V_safe) / U_safe

            mu = ct.mu_g(g, M_u, V_safe, alphas, g_e, g_p, g_f, zD_batch)
            g = g + (dt / substeps) * mu

            if clamp_g:
                g = g.clamp(min=0.0)

        if mode == "full":
            U1, V1, P1, F1 = compute_UVPF(ct, pinn_S, z_D, g)
        else:
            U1, V1, P1, F1 = compute_UVPF_restricted(z_D, g)

        t_list.append((k + 1) * dt)
        z_list.append(z_D)
        U_list.append(U1)
        V_list.append(V1)
        P_list.append(P1)
        F_list.append(F1)

    P_arr = np.asarray(P_list, dtype=float)
    rel_emp = 100.0 * (P_arr / P0)

    return {
        "t": np.asarray(t_list, dtype=float),
        "z_path": np.asarray(z_list, dtype=float),
        "U": np.asarray(U_list, dtype=float),
        "V": np.asarray(V_list, dtype=float),
        "P": P_arr,
        "F": np.asarray(F_list, dtype=float),
        "rel_emp": rel_emp,
        "P0": P0,
        "g_end": g.detach().clone(),   # distribution at t = t_covid, still in D
        "z_end": z_D,
        "mode": mode,
    }


# --------------------------------------------------
# backward-compatible full version
# --------------------------------------------------
@torch.no_grad()
def simulate_common_covid_block(
    ct,
    pinn_S,
    g0_bar,
    z_pre_scalar,
    dt=1e-2,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
):
    return simulate_covid_block(
        ct=ct,
        pinn_S=pinn_S,
        g0_bar=g0_bar,
        z_pre_scalar=z_pre_scalar,
        mode="full",
        g_erg=None,
        dt=dt,
        t_covid=t_covid,
        substeps=substeps,
        clamp_g=clamp_g,
    )


# --------------------------------------------------
# explicit wrappers for clarity
# --------------------------------------------------
@torch.no_grad()
def simulate_full_covid_block(
    ct,
    pinn_S,
    g0_bar,
    z_pre_scalar,
    dt=1e-2,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
):
    return simulate_covid_block(
        ct=ct,
        pinn_S=pinn_S,
        g0_bar=g0_bar,
        z_pre_scalar=z_pre_scalar,
        mode="full",
        g_erg=None,
        dt=dt,
        t_covid=t_covid,
        substeps=substeps,
        clamp_g=clamp_g,
    )


@torch.no_grad()
def simulate_restricted_covid_block(
    ct,
    pinn_S,
    g0_bar,
    z_pre_scalar,
    g_erg=None,
    dt=1e-2,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
):
    return simulate_covid_block(
        ct=ct,
        pinn_S=pinn_S,
        g0_bar=g0_bar,
        z_pre_scalar=z_pre_scalar,
        mode="restricted",
        g_erg=g_erg,
        dt=dt,
        t_covid=t_covid,
        substeps=substeps,
        clamp_g=clamp_g,
    )




def generate_many_LH_paths_from_start(
    ct,
    start_state,
    N_paths,
    T_steps,
    dt,
    seed=None,
):
    """
    Generate many independent 2-state CTMC recovery paths over {L,H}.

    Parameters
    ----------
    start_state : str
        "L" or "H". This is the common initial recovery state for all paths.
    N_paths : int
        Number of independent recovery paths.
    T_steps : int
        Length of each path.
    dt : float
        Time step.
    seed : int or None
        Fixed once for reproducibility of the whole collection.

    Returns
    -------
    z_paths : torch.Tensor
        Shape (N_paths, T_steps), on ct.device, dtype float32.
        Entries are 0.0 for L and 1.0 for H.
    """
    if start_state not in ["L", "H"]:
        raise ValueError("start_state must be 'L' or 'H'")
    if not isinstance(N_paths, int) or N_paths < 1:
        raise ValueError("N_paths must be a positive integer")
    if not isinstance(T_steps, int) or T_steps < 1:
        raise ValueError("T_steps must be a positive integer")
    if dt <= 0:
        raise ValueError("dt must be positive")

    lam_LH = float(ct.lam_LH)
    lam_HL = float(ct.lam_HL)

    p_LH = 1.0 - np.exp(-lam_LH * dt)
    p_HL = 1.0 - np.exp(-lam_HL * dt)

    rng = np.random.default_rng(seed)

    # 0 = L, 1 = H
    s0 = 0 if start_state == "L" else 1

    z_np = np.empty((N_paths, T_steps), dtype=np.float32)
    z_np[:, 0] = float(s0)

    current_states = np.full(N_paths, s0, dtype=np.int64)

    for t in range(1, T_steps):
        u = rng.random(N_paths)

        jump_LH = (current_states == 0) & (u < p_LH)
        jump_HL = (current_states == 1) & (u < p_HL)

        current_states[jump_LH] = 1
        current_states[jump_HL] = 0

        z_np[:, t] = current_states.astype(np.float32)

    return torch.tensor(z_np, device=ct.device, dtype=torch.float32)



@torch.no_grad()
def simulate_full_recovery_many_paths(
    ct,
    pinn_S,
    g_recovery_init,
    z_post_paths,
    P0,
    dt=1e-2,
    substeps=1,
    clamp_g=True,
):
    """
    Full-dynamics recovery simulation for many post-COVID paths.

    Parameters
    ----------
    g_recovery_init : torch.Tensor or array-like
        Common distribution at the end of the COVID block, still in D.
        Shape (1, nx*ny) or (nx*ny,).
    z_post_paths : torch.Tensor or array-like
        Shape (N_paths, T_steps), values in {0,1} for L/H.
        IMPORTANT:
          - column 0 is the recovery-start state at time 0 of recovery
          - total recovery duration is (T_steps - 1) * dt
    P0 : float
        Common pre-COVID ergodic employment normalization.
    dt : float
        Time step.
    substeps : int
        Number of drift substeps per dt.
    clamp_g : bool
        Whether to clamp g to be nonnegative.

    Returns
    -------
    dict with:
      t_rel         : relative recovery time grid, shape (T_steps,)
      rel_emp_paths : shape (N_paths, T_steps)
      rel_emp_mean  : shape (T_steps,)
      P_paths       : shape (N_paths, T_steps)
      P_mean        : shape (T_steps,)
      z_post_paths  : copied numpy array, shape (N_paths, T_steps)
    """
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer. Got: {substeps}")

    z_post_paths_t = torch.as_tensor(z_post_paths, device=ct.device, dtype=torch.float32)
    if z_post_paths_t.ndim != 2:
        raise ValueError("z_post_paths must have shape (N_paths, T_steps)")

    N_paths, T_steps = z_post_paths_t.shape
    if N_paths < 1 or T_steps < 1:
        raise ValueError("z_post_paths must have positive dimensions")

    g_init = torch.as_tensor(g_recovery_init, device=ct.device, dtype=torch.float32).reshape(1, -1)
    sigma_mat_np = ct.sigma_mat.detach().cpu().numpy()

    rel_emp_paths = np.empty((N_paths, T_steps), dtype=float)
    P_paths = np.empty((N_paths, T_steps), dtype=float)

    # simulate each recovery path independently
    for n in range(N_paths):
        g = g_init.clone()
        z_path = z_post_paths_t[n:n+1, :]   # shape (1, T_steps)

        # At recovery time 0, we jump from D to the first LH state if needed
        z0 = float(z_path[0, 0].item())
        z_pre = 2.0  # end of COVID block is in D by construction

        if z_pre != z0:
            z_from0 = int(round(z_pre))
            z_to0 = int(round(z0))
            sigma0 = float(sigma_mat_np[z_from0, z_to0])

            g = (1.0 - sigma0) * g
            if clamp_g:
                g = g.clamp(min=0.0)

        # record post-jump state at recovery time 0
        U0, V0, P0_path, F0 = compute_UVPF(ct, pinn_S, z0, g)
        P_paths[n, 0] = P0_path
        rel_emp_paths[n, 0] = 100.0 * (P0_path / P0)

        # evolve over the recovery path
        for k in range(T_steps - 1):
            z_k = float(z_path[0, k].item())
            z_batch = torch.tensor([[z_k]], device=ct.device, dtype=torch.float32)

            # drift over [t_k, t_{k+1})
            for _ in range(substeps):
                S_grid, alphas = ct.calculate_alphas(pinn_S, z_batch, g)
                g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_grid, alphas, z_batch)

                U_safe = torch.clamp(U_t, min=1e-12)
                V_safe = torch.clamp(V_t, min=1e-12)
                M_u = ct.m(U_safe, V_safe) / U_safe

                mu = ct.mu_g(g, M_u, V_safe, alphas, g_e, g_p, g_f, z_batch)
                g = g + (dt / substeps) * mu

                if clamp_g:
                    g = g.clamp(min=0.0)

            # jump at t_{k+1} if z switches
            z_next = float(z_path[0, k + 1].item())
            if z_next != z_k:
                z_from = int(round(z_k))
                z_to = int(round(z_next))
                sigma_jump = float(sigma_mat_np[z_from, z_to])

                g = (1.0 - sigma_jump) * g
                if clamp_g:
                    g = g.clamp(min=0.0)

            # record post-jump state at t_{k+1}
            U1, V1, P1, F1 = compute_UVPF(ct, pinn_S, z_next, g)
            P_paths[n, k + 1] = P1
            rel_emp_paths[n, k + 1] = 100.0 * (P1 / P0)

    t_rel = np.arange(T_steps, dtype=float) * dt

    return {
        "t_rel": t_rel,
        "rel_emp_paths": rel_emp_paths,
        "rel_emp_mean": rel_emp_paths.mean(axis=0),
        "P_paths": P_paths,
        "P_mean": P_paths.mean(axis=0),
        "z_post_paths": z_post_paths_t.detach().cpu().numpy(),
    }





@torch.no_grad()
def simulate_restricted_recovery_many_paths(
    ct,
    pinn_S,
    g_recovery_init,
    z_post_paths,
    g_erg,
    P0,
    dt=1e-2,
    substeps=1,
    clamp_g=True,
):
    """
    Restricted-dynamics recovery simulation for many post-COVID paths.

    IMPORTANT:
    This follows the same restricted logic as your Cell 5:
      - precompute S_erg[z], alpha_erg[z] at the fixed ergodic g
      - during the recovery, use those fixed objects for the current z
      - current g still evolves over time

    Parameters
    ----------
    g_recovery_init : torch.Tensor or array-like
        Common distribution at the end of the COVID block, still in D.
        Shape (1, nx*ny) or (nx*ny,).
    z_post_paths : torch.Tensor or array-like
        Shape (N_paths, T_steps), values in {0,1} for L/H.
        column 0 is the recovery-start state at time 0 of recovery.
    g_erg : torch.Tensor or array-like
        Ergodic reference distribution used to freeze S and alpha.
    P0 : float
        Common pre-COVID ergodic employment normalization.
    dt : float
        Time step.
    substeps : int
        Number of drift substeps per dt.
    clamp_g : bool
        Whether to clamp g to be nonnegative.

    Returns
    -------
    dict with:
      t_rel         : relative recovery time grid, shape (T_steps,)
      rel_emp_paths : shape (N_paths, T_steps)
      rel_emp_mean  : shape (T_steps,)
      P_paths       : shape (N_paths, T_steps)
      P_mean        : shape (T_steps,)
      z_post_paths  : copied numpy array, shape (N_paths, T_steps)
    """
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer. Got: {substeps}")

    z_post_paths_t = torch.as_tensor(z_post_paths, device=ct.device, dtype=torch.float32)
    if z_post_paths_t.ndim != 2:
        raise ValueError("z_post_paths must have shape (N_paths, T_steps)")

    N_paths, T_steps = z_post_paths_t.shape
    if N_paths < 1 or T_steps < 1:
        raise ValueError("z_post_paths must have positive dimensions")

    g_init = torch.as_tensor(g_recovery_init, device=ct.device, dtype=torch.float32).reshape(1, -1)
    g_erg_t = torch.as_tensor(g_erg, device=ct.device, dtype=torch.float32).reshape(1, -1)

    sigma_mat_np = ct.sigma_mat.detach().cpu().numpy()

    # --------------------------------------------------
    # precompute restricted S and alpha at ergodic g
    # exactly as in your Cell 5
    # --------------------------------------------------
    S_list = []
    alpha_list = []
    for z_val in [0, 1, 2]:
        z_batch = torch.tensor([[float(z_val)]], device=ct.device, dtype=torch.float32)
        S_z, alpha_z = ct.calculate_alphas(pinn_S, z_batch, g_erg_t)
        S_list.append(S_z[0])         # (nx, ny)
        alpha_list.append(alpha_z[0]) # (nx, ny)

    S_erg_stack = torch.stack(S_list, dim=0)           # (3, nx, ny)
    alpha_erg_stack = torch.stack(alpha_list, dim=0)   # (3, nx, ny)

    def compute_UVPF_restricted(z_scalar, g_flat):
        z_idx = int(round(float(z_scalar)))
        z_batch = torch.tensor([[float(z_scalar)]], device=ct.device, dtype=torch.float32)

        S_now = S_erg_stack[z_idx].unsqueeze(0)
        alpha_now = alpha_erg_stack[z_idx].unsqueeze(0)

        g_e, g_u, U, g_p, g_v, V, g_f = ct.marginals_U_V(g_flat, S_now, alpha_now, z_batch)
        P = g_flat.reshape(1, ct.nx, ct.ny).mean(dim=(1, 2), keepdim=True)
        F = P + V

        return float(U.item()), float(V.item()), float(P.item()), float(F.item())

    rel_emp_paths = np.empty((N_paths, T_steps), dtype=float)
    P_paths = np.empty((N_paths, T_steps), dtype=float)

    # --------------------------------------------------
    # simulate each recovery path independently
    # --------------------------------------------------
    for n in range(N_paths):
        g = g_init.clone()
        z_path = z_post_paths_t[n:n+1, :]   # shape (1, T_steps)

        # At recovery time 0, jump from D to the first LH state
        z0 = float(z_path[0, 0].item())
        z_pre = 2.0  # end of COVID block is in D

        if z_pre != z0:
            z_from0 = int(round(z_pre))
            z_to0 = int(round(z0))
            sigma0 = float(sigma_mat_np[z_from0, z_to0])

            g = (1.0 - sigma0) * g
            if clamp_g:
                g = g.clamp(min=0.0)

        # record post-jump state at recovery time 0
        U0, V0, P0_path, F0 = compute_UVPF_restricted(z0, g)
        P_paths[n, 0] = P0_path
        rel_emp_paths[n, 0] = 100.0 * (P0_path / P0)

        # evolve over the recovery path
        for k in range(T_steps - 1):
            z_k = float(z_path[0, k].item())
            z_idx = int(round(z_k))
            z_batch = torch.tensor([[z_k]], device=ct.device, dtype=torch.float32)

            # fixed restricted objects at ergodic g for current z_k
            S_t = S_erg_stack[z_idx].unsqueeze(0)
            alpha_t = alpha_erg_stack[z_idx].unsqueeze(0)

            # drift over [t_k, t_{k+1})
            for _ in range(substeps):
                g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_t, alpha_t, z_batch)

                U_safe = torch.clamp(U_t, min=1e-12)
                V_safe = torch.clamp(V_t, min=1e-12)
                M_u = ct.m(U_safe, V_safe) / U_safe

                mu = ct.mu_g(g, M_u, V_safe, alpha_t, g_e, g_p, g_f, z_batch)
                g = g + (dt / substeps) * mu

                if clamp_g:
                    g = g.clamp(min=0.0)

            # jump at t_{k+1} if z switches
            z_next = float(z_path[0, k + 1].item())
            if z_next != z_k:
                z_from = int(round(z_k))
                z_to = int(round(z_next))
                sigma_jump = float(sigma_mat_np[z_from, z_to])

                g = (1.0 - sigma_jump) * g
                if clamp_g:
                    g = g.clamp(min=0.0)

            # record post-jump state at t_{k+1}
            U1, V1, P1, F1 = compute_UVPF_restricted(z_next, g)
            P_paths[n, k + 1] = P1
            rel_emp_paths[n, k + 1] = 100.0 * (P1 / P0)

    t_rel = np.arange(T_steps, dtype=float) * dt

    return {
        "t_rel": t_rel,
        "rel_emp_paths": rel_emp_paths,
        "rel_emp_mean": rel_emp_paths.mean(axis=0),
        "P_paths": P_paths,
        "P_mean": P_paths.mean(axis=0),
        "z_post_paths": z_post_paths_t.detach().cpu().numpy(),
    }



def build_figure3a_average_result(
    covid_full,
    covid_restricted,
    full_recovery,
    restricted_recovery,
):
    """
    Build the final averaged Figure 3.a objects by stitching:
      - full COVID block + averaged full recovery
      - restricted COVID block + averaged restricted recovery

    Parameters
    ----------
    covid_full : dict
        Output of simulate_full_covid_block(...)
    covid_restricted : dict
        Output of simulate_restricted_covid_block(...)
    full_recovery : dict
        Output of simulate_full_recovery_many_paths(...)
    restricted_recovery : dict
        Output of simulate_restricted_recovery_many_paths(...)

    Returns
    -------
    dict with:
      t                    : common full time grid
      rel_emp_full         : blue line
      rel_emp_restricted   : orange dashed line
      t_covid              : end of COVID block
      covid_line_time      : one dt after start of recovery
      z_post_paths         : recovery-state paths used for both lines
      covid_full_raw       : raw full COVID-block output
      covid_restricted_raw : raw restricted COVID-block output
      full_recovery_raw    : raw full-recovery output
      restricted_recovery_raw : raw restricted-recovery output
    """
    # -----------------------------
    # unpack COVID blocks
    # -----------------------------
    t_full_covid = np.asarray(covid_full["t"], dtype=float)
    t_rest_covid = np.asarray(covid_restricted["t"], dtype=float)

    rel_full_covid = np.asarray(covid_full["rel_emp"], dtype=float)
    rel_rest_covid = np.asarray(covid_restricted["rel_emp"], dtype=float)

    # -----------------------------
    # unpack recovery blocks
    # -----------------------------
    t_rel_full = np.asarray(full_recovery["t_rel"], dtype=float)
    t_rel_rest = np.asarray(restricted_recovery["t_rel"], dtype=float)

    rel_full_rec = np.asarray(full_recovery["rel_emp_mean"], dtype=float)
    rel_rest_rec = np.asarray(restricted_recovery["rel_emp_mean"], dtype=float)

    z_post_full = np.asarray(full_recovery["z_post_paths"])
    z_post_rest = np.asarray(restricted_recovery["z_post_paths"])

    # -----------------------------
    # checks
    # -----------------------------
    if t_full_covid.ndim != 1 or t_rest_covid.ndim != 1:
        raise ValueError("COVID-block time grids must be 1D arrays.")
    if rel_full_covid.ndim != 1 or rel_rest_covid.ndim != 1:
        raise ValueError("COVID-block relative-employment series must be 1D arrays.")

    if len(t_full_covid) != len(rel_full_covid):
        raise ValueError("Full COVID-block time grid and series length mismatch.")
    if len(t_rest_covid) != len(rel_rest_covid):
        raise ValueError("Restricted COVID-block time grid and series length mismatch.")

    if len(t_rel_full) != len(rel_full_rec):
        raise ValueError("Full recovery time grid and mean series length mismatch.")
    if len(t_rel_rest) != len(rel_rest_rec):
        raise ValueError("Restricted recovery time grid and mean series length mismatch.")

    # same deterministic COVID grid for both
    if len(t_full_covid) != len(t_rest_covid):
        raise ValueError("Full and restricted COVID-block grids have different lengths.")
    if not np.allclose(t_full_covid, t_rest_covid):
        raise ValueError("Full and restricted COVID-block grids are not identical.")

    # same recovery grid for both
    if len(t_rel_full) != len(t_rel_rest):
        raise ValueError("Full and restricted recovery grids have different lengths.")
    if not np.allclose(t_rel_full, t_rel_rest):
        raise ValueError("Full and restricted recovery grids are not identical.")

    # same recovery paths for both
    if z_post_full.shape != z_post_rest.shape:
        raise ValueError("Full and restricted recovery path collections have different shapes.")
    if not np.array_equal(z_post_full, z_post_rest):
        raise ValueError("Full and restricted dynamics must use the same recovery paths.")

    # same ergodic normalization should have been used
    P0_full = float(covid_full["P0"])
    P0_rest = float(covid_restricted["P0"])
    if not np.isclose(P0_full, P0_rest):
        raise ValueError("Full and restricted COVID blocks do not use the same P0 normalization.")

    if len(t_full_covid) < 2:
        raise ValueError("COVID block is too short.")

    # -----------------------------
    # timing convention
    # -----------------------------
    dt_common = t_full_covid[1] - t_full_covid[0]
    t_covid = t_full_covid[-1]

    # Recovery arrays start at recovery time 0, i.e. post-jump state at absolute time t_covid.
    # Drop the last COVID point to avoid duplicate x-values at t=t_covid.
    t_covid_used = t_full_covid[:-1]
    rel_full_covid_used = rel_full_covid[:-1]
    rel_rest_covid_used = rel_rest_covid[:-1]

    t_recovery = t_covid + t_rel_full

    t = np.concatenate([t_covid_used, t_recovery])
    rel_emp_full = np.concatenate([rel_full_covid_used, rel_full_rec])
    rel_emp_restricted = np.concatenate([rel_rest_covid_used, rel_rest_rec])

    return {
        "t": t,
        "rel_emp_full": rel_emp_full,
        "rel_emp_restricted": rel_emp_restricted,
        "t_covid": t_covid,
        "covid_line_time": t_covid + dt_common,
        "z_post_paths": z_post_full,
        "P0": P0_full,
        "covid_full_raw": covid_full,
        "covid_restricted_raw": covid_restricted,
        "full_recovery_raw": full_recovery,
        "restricted_recovery_raw": restricted_recovery,
    }



def plot_figure3a_average_many_paths(
    ct,
    pinn_S,
    g0_bar,
    z_pre_scalar,
    z_after_start="H",
    N_recovery_paths=200,
    dt=0.01,
    T_end=2.0,
    t_covid=0.2,
    substeps=1,
    clamp_g=True,
    seed_recovery_paths=123,
):
    """
    Averaged Figure 3.a with consistent full/restricted initialization.

    Logic:
      1) simulate full COVID block  -> covid_full["g_end"]
      2) simulate restricted COVID block -> covid_restricted["g_end"]
      3) generate ONE common collection of many LH recovery paths
      4) run full recovery from covid_full["g_end"]
      5) run restricted recovery from covid_restricted["g_end"]
      6) stitch the two paths separately and plot

    Parameters
    ----------
    g0_bar : array-like
        Pre-COVID ergodic distribution.
    z_pre_scalar : float
        Pre-COVID aggregate state used for the initial jump into D.
    z_after_start : str
        "L" or "H": common initial recovery state at t = t_covid+.
    N_recovery_paths : int
        Number of independent post-COVID recovery paths to average over.
    dt : float
        Time step.
    T_end : float
        Total horizon from t=0.
    t_covid : float
        Length of the COVID block.
    substeps : int
        Number of drift substeps per dt.
    clamp_g : bool
        Whether to clamp g to be nonnegative.
    seed_recovery_paths : int or None
        Fixed once for reproducibility of the whole collection of recovery paths.

    Returns
    -------
    final_out : dict
        Output of build_figure3a_average_result(...)
    """
    if z_after_start not in ["L", "H"]:
        raise ValueError("z_after_start must be 'L' or 'H'")
    if not isinstance(N_recovery_paths, int) or N_recovery_paths < 1:
        raise ValueError("N_recovery_paths must be a positive integer")
    if T_end <= t_covid:
        raise ValueError("T_end must be strictly larger than t_covid")

    # --------------------------------------------------
    # 1) COVID block: full dynamics
    # --------------------------------------------------
    covid_full = simulate_full_covid_block(
        ct=ct,
        pinn_S=pinn_S,
        g0_bar=g0_bar,
        z_pre_scalar=z_pre_scalar,
        dt=dt,
        t_covid=t_covid,
        substeps=substeps,
        clamp_g=clamp_g,
    )

    # --------------------------------------------------
    # 2) COVID block: restricted dynamics
    #    use the same ergodic reference distribution g0_bar
    # --------------------------------------------------
    covid_restricted = simulate_restricted_covid_block(
        ct=ct,
        pinn_S=pinn_S,
        g0_bar=g0_bar,
        z_pre_scalar=z_pre_scalar,
        g_erg=g0_bar,
        dt=dt,
        t_covid=t_covid,
        substeps=substeps,
        clamp_g=clamp_g,
    )

    # --------------------------------------------------
    # 3) Many independent recovery LH paths
    # IMPORTANT:
    # T_steps counts grid points, so duration = (T_steps - 1) * dt.
    # This matches the old Figure 3.a convention, where the last plotted time
    # is typically T_end - dt rather than exactly T_end.
    # --------------------------------------------------
    T_post_steps = int(round((T_end - t_covid) / dt))
    if T_post_steps < 1:
        raise ValueError("Recovery block is too short.")

    z_post_paths = generate_many_LH_paths_from_start(
        ct=ct,
        start_state=z_after_start,
        N_paths=N_recovery_paths,
        T_steps=T_post_steps,
        dt=dt,
        seed=seed_recovery_paths,
    )

    # --------------------------------------------------
    # 4) Full recovery from full COVID endpoint
    # --------------------------------------------------
    full_recovery = simulate_full_recovery_many_paths(
        ct=ct,
        pinn_S=pinn_S,
        g_recovery_init=covid_full["g_end"],
        z_post_paths=z_post_paths,
        P0=covid_full["P0"],
        dt=dt,
        substeps=substeps,
        clamp_g=clamp_g,
    )

    # --------------------------------------------------
    # 5) Restricted recovery from restricted COVID endpoint
    # --------------------------------------------------
    restricted_recovery = simulate_restricted_recovery_many_paths(
        ct=ct,
        pinn_S=pinn_S,
        g_recovery_init=covid_restricted["g_end"],
        z_post_paths=z_post_paths,
        g_erg=g0_bar,
        P0=covid_restricted["P0"],
        dt=dt,
        substeps=substeps,
        clamp_g=clamp_g,
    )

    # --------------------------------------------------
    # 6) Stitch full and restricted paths separately
    # --------------------------------------------------
    final_out = build_figure3a_average_result(
        covid_full=covid_full,
        covid_restricted=covid_restricted,
        full_recovery=full_recovery,
        restricted_recovery=restricted_recovery,
    )


    return final_out



