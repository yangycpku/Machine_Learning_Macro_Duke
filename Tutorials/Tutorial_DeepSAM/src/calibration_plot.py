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



# Defining a function that simulate z_paths before disaster state
def generate_LH_paths(
    N,
    T,
    dt,
    init_probs_LH,
    lam_LH,
    lam_HL,
    device,
    seed=None,
    warn_threshold=0.1,
    as_float=False,
    verbose=False,
):
    """
    Generate N paths of length T for a 2-state CTMC z_t in {0=L, 1=H},
    with transition intensities:
        L -> H : lam_LH
        H -> L : lam_HL

    Time step is dt (in years, e.g. 0.01).
    init_probs_LH: length-2 iterable (p_L, p_H), will be renormalized.
    """

    # Basic checks
    if N <= 0 or T <= 0:
        raise ValueError("N and T must be positive integers.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Convert lambdas and rates
    lam_LH_t = torch.as_tensor(lam_LH, dtype=torch.float32, device=device)
    lam_HL_t = torch.as_tensor(lam_HL, dtype=torch.float32, device=device)

    # Total exit rates q for each state: q[0]=q_L, q[1]=q_H
    q = torch.stack([lam_LH_t, lam_HL_t])  # shape (2,)

    # Check dt * max_rate
    max_q = torch.max(q)
    if max_q * dt > warn_threshold:
        print(
            f"Warning: dt * max_rate = {(max_q * dt).item():.4f} > {warn_threshold}. "
            "One-jump-per-step approximation may be inaccurate."
        )

    # Handle initial probabilities over {L,H}
    init_probs = torch.as_tensor(init_probs_LH, dtype=torch.float32, device=device)
    if init_probs.numel() != 2:
        raise ValueError("init_probs_LH must have length 2 (for L and H).")

    if (init_probs < 0).any():
        raise ValueError("init_probs_LH must be non-negative.")

    # Renormalize just in case they don't sum to 1 exactly
    prob_sum = init_probs.sum()
    if prob_sum <= 0:
        raise ValueError("Sum of init_probs_LH must be positive.")
    init_probs = init_probs / prob_sum

    # Sample initial states z0 ~ {0,1} with given probabilities
    # Use multinomial: shape (N, 2) -> (N, 1) -> (N,)
    z0 = torch.multinomial(init_probs.expand(N, -1), num_samples=1).squeeze(1)  # (N,)
    # z0 entries are 0 (L) or 1 (H)

    # Allocate output tensor
    z = torch.empty((N, T), dtype=torch.long, device=device)
    z[:, 0] = z0

    # Time stepping
    for t in range(1, T):
        prev = z[:, t - 1]  # (N,)

        # q_prev[n] = q[prev[n]]
        q_prev = q[prev]  # (N,)

        # Probability of at least one jump in this dt: 1 - exp(-q_i * dt)
        p_jump = 1.0 - torch.exp(-q_prev * dt)  # (N,)

        # Draw if a jump happens
        u_jump = torch.rand(N, device=device)
        jump_mask = u_jump < p_jump  # (N,) bool

        # Default: stay in same state
        next_state = prev.clone()

        # For jumpers, flip the state: 0 -> 1, 1 -> 0
        # Since states are 0 or 1, we can do 1 - prev
        next_state[jump_mask] = 1 - prev[jump_mask]

        z[:, t] = next_state

        if verbose and (t % max(T // 10, 1) == 0 or t == T - 1):
            share_L = (next_state == 0).float().mean().item()
            share_H = (next_state == 1).float().mean().item()
            avg_p_jump = p_jump.mean().item()
            print(
                f"t={t}/{T-1}: share_L={share_L:.3f}, share_H={share_H:.3f}, "
                f"avg_p_jump={avg_p_jump:.4f}"
            )

    if as_float:
        return z.float()
    return z



@torch.no_grad()
def ergodic_g_LH_ctmc(
    ct,
    pinn_S,
    g_init=None,
    N_paths=200,
    dt=1e-2,
    T_end=30.0,
    burn_in=10.0,
    record_interval=10,
    substeps=1,
    clamp_g=True,
    seed=123,
    init_probs_LH=None,
):
    """
    Ergodic g0 from pre-COVID 2-state CTMC over {L,H} only.
    Uses:
      - generate_LH_paths(...)
      - ct.calculate_alphas(...)
      - ct.marginals_U_V(...)
      - ct.mu_g(...)
    Returns:
      g_bar      : averaged ergodic g, shape (1, nx*ny)
      last_stats : average final U,V,P,F across simulated paths
      z_paths    : simulated 2-state z paths
    """
    device = ct.device
    T = int(round(T_end / dt))
    burn_steps = int(round(burn_in / dt))

    if init_probs_LH is None:
        init_probs_LH = [ct.pi_L, ct.pi_H]  # generate_LH_paths renormalizes internally

    # ---------- initial g ----------
    if g_init is None:
        def get_g0_from_ct():
            for attr in ["gm_low", "gm_L", "gm_ss", "gm_low_delta", "gm_ss_delta", "gm_high_delta", "gm_dis_delta"]:
                if hasattr(ct, attr):
                    g = getattr(ct, attr)
                    if g is not None:
                        return g
            if hasattr(ct, "path"):
                for fname in ["gm_low_delta.npy", "gm_low.npy", "gm_high_delta.npy", "gm_dis_delta.npy", "gm_ss.npy"]:
                    fpath = os.path.join(ct.path, fname)
                    if os.path.exists(fpath):
                        return np.load(fpath)
            raise RuntimeError("Could not find initial g. Provide g_init or run ct.solve_steady_state().")

        g_init = get_g0_from_ct()

    g_init = torch.as_tensor(g_init, device=device, dtype=torch.float32).reshape(1, -1)
    g = g_init.repeat(N_paths, 1).clone()

    # ---------- 2-state z paths ----------
    z_paths = generate_LH_paths(
        N=N_paths,
        T=T,
        dt=dt,
        init_probs_LH=init_probs_LH,
        lam_LH=ct.lam_LH,
        lam_HL=ct.lam_HL,
        device=device,
        seed=seed,
        as_float=False,
        verbose=False,
    )  # shape (N_paths, T), values in {0,1}

    sigma_mat = ct.sigma_mat.to(device)

    sum_g = torch.zeros_like(g_init)
    count = 0

    # ---------- simulation ----------
    for k in range(T):
        z_batch = z_paths[:, k].float().reshape(-1, 1)

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

        # exact jump at t_{k+1} if z switches
        if k < T - 1:
            z_from = z_paths[:, k].long()
            z_to   = z_paths[:, k + 1].long()
            jump_mask = (z_to != z_from)

            if jump_mask.any():
                sigma_jump = sigma_mat[z_from, z_to].reshape(-1, 1)
                factor = torch.ones((N_paths, 1), device=device, dtype=torch.float32)
                factor[jump_mask] = 1.0 - sigma_jump[jump_mask]
                g = factor * g

                if clamp_g:
                    g = g.clamp(min=0.0)

        # record post-burn-in
        if (k >= burn_steps) and (k % record_interval == 0):
            sum_g += g.sum(dim=0, keepdim=True)
            count += N_paths

    if count == 0:
        raise RuntimeError("No samples collected for averaging. Increase T_end or reduce burn_in/record_interval.")

    g_bar = (sum_g / count).detach()

    # ---------- final average stats ----------
    z_last = z_paths[:, -1].float().reshape(-1, 1)
    S_grid, alphas = ct.calculate_alphas(pinn_S, z_last, g)
    g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_grid, alphas, z_last)
    P_t = g.reshape(N_paths, ct.nx, ct.ny).mean(dim=(1, 2), keepdim=True)
    F_t = P_t + V_t

    last_stats = {
        "U": float(U_t.mean().item()),
        "V": float(V_t.mean().item()),
        "P": float(P_t.mean().item()),
        "F": float(F_t.mean().item()),
    }

    return g_bar, last_stats, z_paths



@torch.no_grad()
def simulate_disaster_0p2_for_figure1(
    ct,
    pinn_S,
    g0_paths,
    z_pre_paths,
    dt=1e-2,
    T_shock=0.2,
    substeps=1,
    clamp_g=True,
):
    """
    Simulate the economy for 0.2 years in disaster, with an immediate jump at t=0
    from z_pre in {L,H} to D.

    Inputs
    ------
    g0_paths   : shape (N, nx*ny) or (1, nx*ny)
                 pre-COVID pathwise initial g
    z_pre_paths: shape (N,) with values 0/1 for L/H

    Returns
    -------
    out : dict with
        g_pre, g_post,
        worker_emp_pre_mean, worker_emp_post_mean,
        worker_unemp_pre_mean, worker_unemp_post_mean,
        worker_unemp_rate_pre_mean, worker_unemp_rate_post_mean,
        worker_emp_drop_mean, worker_emp_drop_pct_mean,
        firm_emp_pre_mean, firm_emp_post_mean,
        firm_vac_pre_mean, firm_vac_post_mean,
        firm_emp_rate_pre_mean, firm_emp_rate_post_mean,
        firm_vac_rate_pre_mean, firm_vac_rate_post_mean,
        firm_emp_drop_mean, firm_emp_drop_pct_mean
    """
    device = ct.device
    nx, ny = ct.nx, ct.ny
    D_state = 2
    T_steps = int(round(T_shock / dt))

    g0_paths = torch.as_tensor(g0_paths, device=device, dtype=torch.float32)
    z_pre_paths = torch.as_tensor(z_pre_paths, device=device, dtype=torch.long).reshape(-1)

    if g0_paths.ndim == 1:
        g0_paths = g0_paths.reshape(1, -1)

    N = z_pre_paths.shape[0]

    if g0_paths.shape[0] == 1 and N > 1:
        g0_paths = g0_paths.repeat(N, 1)

    if g0_paths.shape[0] != N:
        raise ValueError("g0_paths and z_pre_paths must have the same number of paths.")

    gw = ct.gw.to(device).reshape(1, nx)   # shape (1, nx)

    # ---------- pre-shock objects ----------
    z_pre_batch = z_pre_paths.float().reshape(-1, 1)
    S_pre, alphas_pre = ct.calculate_alphas(pinn_S, z_pre_batch, g0_paths)
    g_e_pre, g_u_pre, U_pre, g_p_pre, g_v_pre, V_pre, g_f_pre = ct.marginals_U_V(
        g0_paths, S_pre, alphas_pre, z_pre_batch
    )

    worker_emp_pre = g0_paths.reshape(N, nx, ny).mean(dim=2)
    worker_unemp_pre = gw - worker_emp_pre
    worker_unemp_rate_pre = worker_unemp_pre / gw

    firm_emp_pre = g_p_pre
    firm_vac_pre = g_v_pre
    firm_emp_rate_pre = g_p_pre / g_f_pre
    firm_vac_rate_pre = g_v_pre / g_f_pre

    # ---------- t=0 jump to disaster ----------
    sigma_jump = ct.sigma_mat[z_pre_paths, torch.full_like(z_pre_paths, D_state)].reshape(-1, 1)
    g = (1.0 - sigma_jump) * g0_paths

    if clamp_g:
        g = g.clamp(min=0.0)

    # ---------- drift in disaster for 0.2 years ----------
    zD_batch = torch.full((N, 1), float(D_state), device=device)

    for _ in range(T_steps):
        for _ in range(substeps):
            S_grid, alphas = ct.calculate_alphas(pinn_S, zD_batch, g)
            g_e, g_u, U_t, g_p, g_v, V_t, g_f = ct.marginals_U_V(g, S_grid, alphas, zD_batch)

            U_safe = torch.clamp(U_t, min=1e-12)
            V_safe = torch.clamp(V_t, min=1e-12)
            M_u = ct.m(U_safe, V_safe) / U_safe

            mu = ct.mu_g(g, M_u, V_safe, alphas, g_e, g_p, g_f, zD_batch)
            g = g + (dt / substeps) * mu

            if clamp_g:
                g = g.clamp(min=0.0)

    g_post = g

    # ---------- post-shock objects at t=0.2 ----------
    S_post, alphas_post = ct.calculate_alphas(pinn_S, zD_batch, g_post)
    g_e_post, g_u_post, U_post, g_p_post, g_v_post, V_post, g_f_post = ct.marginals_U_V(
        g_post, S_post, alphas_post, zD_batch
    )

    worker_emp_post = g_post.reshape(N, nx, ny).mean(dim=2)
    worker_unemp_post = gw - worker_emp_post
    worker_unemp_rate_post = worker_unemp_post / gw

    firm_emp_post = g_p_post
    firm_vac_post = g_v_post
    firm_emp_rate_post = g_p_post / g_f_post
    firm_vac_rate_post = g_v_post / g_f_post

    # ---------- pathwise changes ----------
    worker_emp_drop = worker_emp_pre - worker_emp_post
    worker_emp_drop_pct = worker_emp_drop / worker_emp_pre

    firm_emp_drop = firm_emp_pre - firm_emp_post
    firm_emp_drop_pct = firm_emp_drop / firm_emp_pre

    # ---------- averages across paths ----------
    out = {
        "g_pre": g0_paths.detach().cpu(),
        "g_post": g_post.detach().cpu(),
        "z_pre": z_pre_paths.detach().cpu(),

        "worker_emp_pre_mean": worker_emp_pre.mean(dim=0).detach().cpu(),
        "worker_emp_post_mean": worker_emp_post.mean(dim=0).detach().cpu(),
        "worker_unemp_pre_mean": worker_unemp_pre.mean(dim=0).detach().cpu(),
        "worker_unemp_post_mean": worker_unemp_post.mean(dim=0).detach().cpu(),
        "worker_unemp_rate_pre_mean": worker_unemp_rate_pre.mean(dim=0).detach().cpu(),
        "worker_unemp_rate_post_mean": worker_unemp_rate_post.mean(dim=0).detach().cpu(),
        "worker_emp_drop_mean": worker_emp_drop.mean(dim=0).detach().cpu(),
        "worker_emp_drop_pct_mean": worker_emp_drop_pct.mean(dim=0).detach().cpu(),

        "firm_emp_pre_mean": firm_emp_pre.mean(dim=0).detach().cpu(),
        "firm_emp_post_mean": firm_emp_post.mean(dim=0).detach().cpu(),
        "firm_vac_pre_mean": firm_vac_pre.mean(dim=0).detach().cpu(),
        "firm_vac_post_mean": firm_vac_post.mean(dim=0).detach().cpu(),
        "firm_emp_rate_pre_mean": firm_emp_rate_pre.mean(dim=0).detach().cpu(),
        "firm_emp_rate_post_mean": firm_emp_rate_post.mean(dim=0).detach().cpu(),
        "firm_vac_rate_pre_mean": firm_vac_rate_pre.mean(dim=0).detach().cpu(),
        "firm_vac_rate_post_mean": firm_vac_rate_post.mean(dim=0).detach().cpu(),
        "firm_emp_drop_mean": firm_emp_drop.mean(dim=0).detach().cpu(),
        "firm_emp_drop_pct_mean": firm_emp_drop_pct.mean(dim=0).detach().cpu(),
    }

    return out









