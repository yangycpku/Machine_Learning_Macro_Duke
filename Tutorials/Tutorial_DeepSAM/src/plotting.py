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

import covid_shock_plot as covplot
import calibration_plot as calplot



def plot_relative_employment_2a(res_fig3a, res_fig3a_orange):
        # --------------------------------------------------
        # dates
        # --------------------------------------------------
        start_date = pd.Timestamp("2020-02-01")
        dates = start_date + pd.to_timedelta(res_fig3a["t"] * 365.25, unit="D")

        # --------------------------------------------------
        # infer dt from the simulated time grid
        # --------------------------------------------------
        t_grid = np.asarray(res_fig3a["t"], dtype=float)
        if len(t_grid) < 2:
            raise ValueError("Time grid is too short to infer dt.")
        dt_plot = t_grid[1] - t_grid[0]

        # --------------------------------------------------
        # infer end of COVID block from realized z-path
        # D = 2 in your code
        # line should be one dt after the last D point
        # --------------------------------------------------
        z_path = np.asarray(res_fig3a["z_path"], dtype=float)
        idx_after_covid = np.where(z_path != 2.0)[0]

        if len(idx_after_covid) == 0:
            covid_line_date = None
        else:
            first_after_covid_idx = idx_after_covid[0]
            covid_line_time = t_grid[first_after_covid_idx] + dt_plot
            covid_line_date = start_date + pd.to_timedelta(covid_line_time * 365.25, unit="D")

        # --------------------------------------------------
        # plot
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6.5))

        # place the event line exactly at the trough (phase change)
        event_line_date = dates[np.argmin(res_fig3a["rel_emp"])]

        # margin before the series starts (wider shaded band on the left)
        left_pad = pd.Timedelta(days=35)
        x_left = dates.min() - left_pad

        # shaded recession band: from the padded left edge -> event line
        ax.axvspan(x_left, event_line_date, color="indianred", alpha=0.10, zorder=0)

        ax.plot(dates, res_fig3a["rel_emp"] / 100.0, lw=2.5, label="Full dynamics")
        ax.plot(
            dates,
            res_fig3a_orange["rel_emp"] / 100.0,
            lw=2.5,
            linestyle="--",
            label="Restricted dynamics"
        )

        # 100% reference line: faint gray dotted
        ax.axhline(1.0, lw=1.2, linestyle=":", color="gray", alpha=0.7)

        # vertical event line: thin dashed red (at the trough)
        ax.axvline(event_line_date, lw=1.5, linestyle="--", color="red", alpha=0.85)

        # "Recession" label centered in the band
        ax.text(
            x_left + (event_line_date - x_left) * 0.5,
            0.985,
            "Recession",
            color="indianred",
            fontsize=13,
            fontstyle="italic",
            ha="center",
            va="top",
        )

        # x-limits: include the pre-series shaded margin
        ax.set_xlim(x_left, dates.max())

        # labels
        ax.set_xlabel("Date", fontsize=25)
        ax.set_ylabel("Relative Employment", fontsize=25)

        # percent axis
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

        # date ticks
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # styling
        ax.tick_params(axis="both", labelsize=20, width=1.8, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)

        ax.legend(frameon=True, fontsize=17, loc="lower right")

        plt.tight_layout()
        plt.show()



def plot_relative_endogenous_drift(ct, res_fig3a):
    out = res_fig3a["out_raw"]

    # realized -varsigma(z_t)
    varsigma_np = ct.varsigma_s.detach().cpu().numpy()
    z_idx = np.rint(out["z"]).astype(int)
    minus_varsigma_t = -varsigma_np[z_idx]

    # calendar dates
    start_date = pd.Timestamp("2020-02-01")
    dates = start_date + pd.to_timedelta(out["t"] * 365.25, unit="D")

    # recession / event date: same date as the previous plots (hardcoded)
    event_line_date = pd.Timestamp("2020-04-01")
    recovery_line_date = pd.Timestamp("2020-04-17")

    series = out["muF_over_F"] - minus_varsigma_t

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=400)

    # margin before the series starts (wider shaded band on the left)
    left_pad = pd.Timedelta(days=35)
    x_left = dates.min() - left_pad
    x_right = dates.max()

    # shaded recession band
    ax.axvspan(x_left, event_line_date, color="indianred", alpha=0.10, zorder=0)

    # shaded expansion band
    ax.axvspan(event_line_date, x_right, color="seagreen", alpha=0.07, zorder=0)

    ax.plot(dates, series, lw=2.5, color="black")

    # zero reference line: faint gray dotted
    ax.axhline(0.0, lw=1.2, linestyle=":", color="gray", alpha=0.7)

    # vertical event line: thin dashed red
    ax.axvline(event_line_date, lw=1.5, linestyle="--", color="red", alpha=0.85)
    # vertical event line: thin dashed red
    ax.axvline(recovery_line_date, lw=1.5, linestyle="--", color="seagreen", alpha=0.85)

    # y-limits (set before computing label y so labels sit consistently)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)
    y_text = ymin + (ymax - ymin) * 1.01

    # "Recession" label centered in the band
    ax.text(
        x_left + (event_line_date - x_left) * 0.5,
        y_text,
        "Recession",
        color="indianred",
        fontsize=13,
        fontstyle="italic",
        ha="center",
        va="top",
    )

    # "Expansion" label centered in the band
    ax.text(
        event_line_date + (x_right - event_line_date) * 0.5,
        y_text,
        "Expansion",
        color="seagreen",
        fontsize=13,
        fontstyle="italic",
        ha="center",
        va="top",
    )

    # x-limits
    ax.set_xlim(x_left, x_right)

    # labels and title
    ax.set_ylabel(r"Relative endogenous drift", fontsize=25)
    ax.set_xlabel("Date", fontsize=25)
    # ax.set_title(r"$\mu^F/F + \varsigma(z_t)$", fontsize=22, pad=16)

    # date ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    gap_shift = pd.Timedelta(days=1)

    nan_mask = np.isnan(series)
    in_gap = False
    for i in range(len(series)):
        if nan_mask[i] and not in_gap:
            gs_i = i
            in_gap = True
        elif not nan_mask[i] and in_gap:
            if gs_i > 0:
                ax.axvspan(dates[gs_i] - gap_shift, dates[i] - gap_shift,
                        color="dimgray", alpha=0.20, zorder=0)
            in_gap = False

    # styling
    ax.tick_params(axis="both", labelsize=20, width=1.8, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)

    plt.tight_layout()
    plt.show()



def plot_relative_endogenous_jump(ct, res_fig3a, z_pre_scalar):
    out = res_fig3a["out_raw"]
    sigma_mat_np = ct.sigma_mat.detach().cpu().numpy()

    # local reference series around every jump
    ref_sigma = covplot.local_sigma_ref_series(
        out,
        sigma_mat_np,
        z_pre=z_pre_scalar,
        window_steps=2
    )

    # calendar dates
    start_date = pd.Timestamp("2020-02-01")
    dates = start_date + pd.to_timedelta(out["t"] * 365.25, unit="D")

    # event dates
    event_line_date = pd.Timestamp("2020-04-01")
    recovery_line_date = pd.Timestamp("2020-04-17")

    # jump points only
    sig = np.asarray(out["sigmaF_over_F"], dtype=float)
    ref = np.asarray(ref_sigma, dtype=float)
    idx = np.where(~np.isnan(sig))[0]

    # combined quantity at jumps: gamma^F/F + sigma  (ref_sigma holds -sigma, so add sigma = subtract ref)
    combined = sig[idx] - ref[idx]

    # clamp small negative values (|.| < threshold) up to 0; leave positives and large negatives as-is
    neg_threshold = 1e-3
    combined = np.where((combined < 0) & (np.abs(combined) < neg_threshold), 0.0, combined)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=400)

    # margin / band edges
    left_pad = pd.Timedelta(days=35)
    x_left = dates.min() - left_pad
    x_right = dates.max()

    # shaded recession band
    ax.axvspan(x_left, event_line_date, color="indianred", alpha=0.10, zorder=0)

    # shaded expansion band
    ax.axvspan(event_line_date, x_right, color="seagreen", alpha=0.07, zorder=0)

    # gray band between recession and recovery lines
    ax.axvspan(event_line_date, recovery_line_date, color="dimgray", alpha=0.20, zorder=0)

    # x-positions for the jumps; force the 2nd jump to the midpoint of red/green lines
    jump_dates = list(dates[idx])
    mid_date = event_line_date + (recovery_line_date - event_line_date) / 2
    if len(jump_dates) >= 2:
        jump_dates[1] = mid_date

    # combined jump series as stems
    if idx.size > 0:
        markerline, stemlines, baseline = ax.stem(
            jump_dates, combined,
            basefmt=" ",
            linefmt="-",
            markerfmt="o",
        )
        plt.setp(stemlines, linewidth=1.8, color="C0")
        plt.setp(markerline, markersize=6, color="C0")

    # zero reference line
    ax.axhline(0.0, lw=1.2, linestyle=":", color="gray", alpha=0.7)

    # event lines
    ax.axvline(event_line_date, lw=1.5, linestyle="--", color="red", alpha=0.85)
    ax.axvline(recovery_line_date, lw=1.5, linestyle="--", color="seagreen", alpha=0.85)

    # y-limits and labels
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.05)
    y_text = ymin + (ymax - ymin) * 1.01

    # "Recession" label centered in the band
    ax.text(
        x_left + (event_line_date - x_left) * 0.5,
        y_text,
        "Recession",
        color="indianred",
        fontsize=13,
        fontstyle="italic",
        ha="center",
        va="top",
    )

    # "Expansion" label centered in the band
    ax.text(
        event_line_date + (x_right - event_line_date) * 0.5,
        y_text,
        "Expansion",
        color="seagreen",
        fontsize=13,
        fontstyle="italic",
        ha="center",
        va="top",
    )

    # x-limits
    ax.set_xlim(x_left, x_right)

    # labels
    ax.set_ylabel(r"Relative endogenous jump", fontsize=25)
    ax.set_xlabel("Date", fontsize=25)

    # date ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # styling
    ax.tick_params(axis="both", labelsize=20, width=1.8, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)

    plt.tight_layout()
    plt.show()




def plot_relative_employment_many(final_out):
    start_date = pd.Timestamp("2020-02-01")
    dates = start_date + pd.to_timedelta(final_out["t"] * 365.25, unit="D")
    covid_line_date = start_date + pd.to_timedelta(final_out["covid_line_time"] * 365.25, unit="D")

    # shift the event line slightly left to sit at the trough / phase change
    line_shift = pd.Timedelta(days=3)
    event_line_date = covid_line_date - line_shift

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=400)

    # larger margin before the series starts (wider shaded band on the left)
    left_pad = pd.Timedelta(days=35)
    x_left = dates.min() - left_pad

    # shaded recession band: from the padded left edge -> event line
    ax.axvspan(x_left, event_line_date, color="indianred", alpha=0.10, zorder=0)

    # lines as before: solid full, dashed restricted
    ax.plot(dates, final_out["rel_emp_full"] / 100.0, lw=2.6, label="Full dynamics")
    ax.plot(
        dates,
        final_out["rel_emp_restricted"] / 100.0,
        lw=2.6,
        linestyle="--",
        label="Restricted dynamics"
    )

    # 100% reference line: faint gray dotted
    ax.axhline(1.0, lw=1.8, linestyle="-", color="gray", alpha=0.7)

    # vertical event line: thin dashed red (shifted left)
    ax.axvline(event_line_date, lw=1.8, linestyle="--", color="red", alpha=0.85)

    # "Recession" label centered in the band
    ax.text(
        x_left + (event_line_date - x_left) * 0.5,
        0.965,
        "Recession",
        color="indianred",
        fontsize=13,
        fontstyle="italic",
        ha="center",
        va="top",
    )

    # x-limits: include the pre-series shaded margin
    ax.set_xlim(x_left, dates.max())

    # labels
    ax.set_xlabel("Date", fontsize=25)
    ax.set_ylabel("Relative Employment", fontsize=25)

    # percent axis
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    # date ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # styling
    ax.tick_params(axis="both", labelsize=20, width=2.2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    ax.legend(frameon=True, fontsize=17, loc="lower right")

    plt.tight_layout()
    plt.show()







# --------------------------------------------------
# helpers
# --------------------------------------------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def reshape_g_matrix(g_like, nx, ny):
    g_np = to_numpy(g_like)

    if g_np.shape == (nx, ny):
        return g_np

    if g_np.ndim == 1 and g_np.size == nx * ny:
        return g_np.reshape(nx, ny)

    if g_np.ndim == 2 and g_np.shape == (1, nx * ny):
        return g_np.reshape(nx, ny)

    if g_np.ndim == 2 and g_np.shape == (nx * ny, 1):
        return g_np.reshape(nx, ny)

    raise ValueError(f"Unexpected shape for g: {g_np.shape}")

def z_label_to_idx(z_label):
    if z_label == "L":
        return 0
    elif z_label == "H":
        return 1
    else:
        raise ValueError("z_ergodic_for_alpha must be 'L' or 'H'.")

def symmetric_limits(arr):
    m = float(np.max(np.abs(arr)))
    if np.isclose(m, 0.0):
        m = 1e-8
    return -m, m


# --------------------------------------------------
# compute figure-4 objects only (no plotting)
# --------------------------------------------------
def compute_figure4_extended_objects(
    ct,
    pinn_S,
    z_ergodic_for_alpha="L",
    N_paths=200,
    dt=0.01,
    T_end=30.0,
    burn_in=10.0,
    record_interval=10,
    substeps=1,
    clamp_g=True,
    seed=123,
    T_shock=0.2,
):
    @torch.no_grad()
    def _run():
        pinn_S.eval()

        nx, ny = ct.nx, ct.ny
        z_alpha_idx = z_label_to_idx(z_ergodic_for_alpha)
        z_D_idx = 2

        # ------------------------------------------
        # 1) Pre-COVID ergodic distribution g^ergodic
        # ------------------------------------------
        g0_bar, stats0, z_paths_noD = calplot.ergodic_g_LH_ctmc(
            ct=ct,
            pinn_S=pinn_S,
            g_init=None,
            N_paths=N_paths,
            dt=dt,
            T_end=T_end,
            burn_in=burn_in,
            record_interval=record_interval,
            substeps=substeps,
            clamp_g=clamp_g,
            seed=seed,
        )

        g_erg_mat = reshape_g_matrix(g0_bar, nx, ny)
        g_erg_flat = torch.as_tensor(g_erg_mat.reshape(1, -1), device=ct.device, dtype=torch.float32)

        # ------------------------------------------
        # 2) End-of-COVID distribution g^COVID
        #    (before recovery jump)
        # ------------------------------------------
        z_pre_paths = z_paths_noD[:, -1]

        fig4_tmp = calplot.simulate_disaster_0p2_for_figure1(
            ct=ct,
            pinn_S=pinn_S,
            g0_paths=g0_bar,
            z_pre_paths=z_pre_paths,
            dt=dt,
            T_shock=T_shock,
            substeps=substeps,
            clamp_g=clamp_g,
        )

        g_covid_flat_mean = to_numpy(fig4_tmp["g_post"]).mean(axis=0)
        g_covid_mat = reshape_g_matrix(g_covid_flat_mean, nx, ny)
        g_covid_flat = torch.as_tensor(g_covid_mat.reshape(1, -1), device=ct.device, dtype=torch.float32)

        # ------------------------------------------
        # 3) Current recovery distribution:
        #    right after jump from D to chosen recovery state
        # ------------------------------------------
        sigma_D_to_rec = float(to_numpy(ct.sigma_mat)[z_D_idx, z_alpha_idx])
        g_recovery0_flat = (1.0 - sigma_D_to_rec) * g_covid_flat
        if clamp_g:
            g_recovery0_flat = g_recovery0_flat.clamp(min=0.0)

        g_recovery0_mat = reshape_g_matrix(g_recovery0_flat, nx, ny)

        # ------------------------------------------
        # 4) Alpha at ergodic state and recovery-start state
        # ------------------------------------------
        z_alpha_batch = torch.tensor([[float(z_alpha_idx)]], device=ct.device, dtype=torch.float32)

        _, alpha_erg = ct.calculate_alphas(pinn_S, z_alpha_batch, g_erg_flat)
        S_recovery, alpha_recovery = ct.calculate_alphas(pinn_S, z_alpha_batch, g_recovery0_flat)

        alpha_erg_mat = to_numpy(alpha_erg)[0]
        alpha_recovery_mat = to_numpy(alpha_recovery)[0]

        # ------------------------------------------
        # 5) Current g^u(x), g^v(y) at start of recovery
        # ------------------------------------------
        g_e_rec, g_u_rec, U_rec, g_p_rec, g_v_rec, V_rec, g_f_rec = ct.marginals_U_V(
            g_recovery0_flat, S_recovery, alpha_recovery, z_alpha_batch
        )

        g_u_rec_vec = to_numpy(g_u_rec)[0]
        g_v_rec_vec = to_numpy(g_v_rec)[0]

        # ------------------------------------------
        # 6) Relative changes and useful differences
        # ------------------------------------------
        eps = 1e-12

        g_rel_change_pct = 100.0 * (g_covid_mat - g_erg_mat) / np.maximum(g_erg_mat, eps)
        alpha_rel_change_pct = 100.0 * (alpha_recovery_mat - alpha_erg_mat) / np.maximum(alpha_erg_mat, eps)
        alpha_diff = alpha_recovery_mat - alpha_erg_mat
        weighted_alpha_diff = alpha_diff * g_u_rec_vec[:, None] * g_v_rec_vec[None, :]

        return {
            "nx": nx,
            "ny": ny,
            "g_ergodic": g_erg_mat,
            "g_covid": g_covid_mat,
            "g_recovery0": g_recovery0_mat,
            "alpha_ergodic": alpha_erg_mat,
            "alpha_recovery": alpha_recovery_mat,
            "g_rel_change_pct": g_rel_change_pct,
            "alpha_rel_change_pct": alpha_rel_change_pct,
            "alpha_diff": alpha_diff,
            "g_u_recovery": g_u_rec_vec,
            "g_v_recovery": g_v_rec_vec,
            "weighted_alpha_diff": weighted_alpha_diff,
            "z_ergodic_for_alpha": z_ergodic_for_alpha,
        }

    return _run()


# --------------------------------------------------
# plot one single panel
# --------------------------------------------------
# from matplotlib.ticker import FuncFormatter

def _plot_single_panel(
    data,
    title,
    cmap,
    dpi=180,
    panel_ratio=961/543,
    fig_height=4.8,
    colorbar_mode=None,   # None | "percent_1" | "percent_2" | "float4" | "sci3"
    show_values=False,
    value_fmt="{:.3f}",
):
    vmin = vmax = None
    if cmap == "coolwarm":
        vmin, vmax = symmetric_limits(data)

    fig_width = panel_ratio * fig_height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    img = ax.imshow(
        data.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(r"Worker type $x$", fontsize=16)
    ax.set_ylabel(r"Firm type $y$", fontsize=16)

    cbar = fig.colorbar(img, ax=ax, shrink=0.9)

    # only format ticks; do NOT reset them manually
    if colorbar_mode == "percent_1":
        cbar.formatter = FuncFormatter(lambda v, pos: f"{v:.1f}%")
        cbar.update_ticks()
    elif colorbar_mode == "percent_2":
        cbar.formatter = FuncFormatter(lambda v, pos: f"{v:.2f}%")
        cbar.update_ticks()
    elif colorbar_mode == "float4":
        cbar.formatter = FuncFormatter(lambda v, pos: f"{v:.4f}")
        cbar.update_ticks()
    elif colorbar_mode == "sci3":
        cbar.formatter = FuncFormatter(lambda v, pos: f"{v:.3e}")
        cbar.update_ticks()

    if show_values:
        nx, ny = data.shape
        x_centers = (np.arange(nx) + 0.5) / nx
        y_centers = (np.arange(ny) + 0.5) / ny
        maxabs = float(np.max(np.abs(data)))

        for i in range(nx):
            for j in range(ny):
                val = data[i, j]
                txt = value_fmt.format(val)
                color = "white" if (maxabs > 0 and abs(val) > 0.55 * maxabs) else "black"
                ax.text(
                    x_centers[i],
                    y_centers[j],
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )

    plt.tight_layout()
    plt.show()


def plot_figure4_panels_separately(
    ct,
    pinn_S,
    z_ergodic_for_alpha="L",
    N_paths=200,
    dt=0.01,
    T_end=30.0,
    burn_in=10.0,
    record_interval=10,
    substeps=1,
    clamp_g=True,
    seed=123,
    T_shock=0.2,
    dpi=180,
    panel_ratio=961/543,
    fig_height=4.8,
    panels=("a", "b", "c", "d", "e", "f"),
    show_values_in_f=True,
):
    out = compute_figure4_extended_objects(
        ct=ct,
        pinn_S=pinn_S,
        z_ergodic_for_alpha=z_ergodic_for_alpha,
        N_paths=N_paths,
        dt=dt,
        T_end=T_end,
        burn_in=burn_in,
        record_interval=record_interval,
        substeps=substeps,
        clamp_g=clamp_g,
        seed=seed,
        T_shock=T_shock,
    )

    zlab = out["z_ergodic_for_alpha"]

    if "a" in panels:
        _plot_single_panel(
            data=out["g_ergodic"],
            title=r"(a) $g^{\mathrm{ergodic}}(x,y)$ at ergodic state",
            cmap="viridis",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode=None,
        )

    if "b" in panels:
        _plot_single_panel(
            data=out["g_rel_change_pct"],
            title=r"(b) $\dfrac{g^{\mathrm{COVID}}(x,y)-g^{\mathrm{ergodic}}(x,y)}{g^{\mathrm{ergodic}}(x,y)}$",
            cmap="coolwarm",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode="percent_1",
        )

    if "c" in panels:
        _plot_single_panel(
            data=out["alpha_ergodic"],
            title=rf"(c) $\alpha(x,y,z={zlab},g^{{\mathrm{{ergodic}}}})$ at ergodic state",
            cmap="viridis",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode=None,
        )

    if "d" in panels:
        _plot_single_panel(
            data=out["alpha_rel_change_pct"],
            title=rf"(d) $\dfrac{{\alpha(x,y,z={zlab},g^{{\mathrm{{rec}}}})-\alpha(x,y,z={zlab},g^{{\mathrm{{ergodic}}}})}}{{\alpha(x,y,z={zlab},g^{{\mathrm{{ergodic}}}})}}$",
            cmap="coolwarm",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode="percent_1",
        )

    if "e" in panels:
        _plot_single_panel(
            data=out["alpha_diff"],
            title=rf"(e) $\alpha(x,y,z={zlab},g^{{\mathrm{{rec}}}})-\alpha(x,y,z={zlab},g^{{\mathrm{{ergodic}}}})$",
            cmap="coolwarm",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode="float4",
        )

    if "f" in panels:
        _plot_single_panel(
            data=out["weighted_alpha_diff"],
            title=rf"(f) $(\alpha^{{\mathrm{{rec}}}}-\alpha^{{\mathrm{{erg}}}})\,g^u_{{\mathrm{{rec}}}}(x)\,g^v_{{\mathrm{{rec}}}}(y)$ at $z={zlab}$",
            cmap="coolwarm",
            dpi=dpi,
            panel_ratio=panel_ratio,
            fig_height=fig_height,
            colorbar_mode="sci3",
            show_values=show_values_in_f,
            value_fmt="{:.3e}",
        )

    return out



def freeze_S_batches(loader, seed=1234, max_batches=None):
    # Capture the concrete S batches once (on CPU), reproducibly.
    S_list = []
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for i, batch in enumerate(loader):
            S = batch[0] if isinstance(batch, (tuple, list)) else batch
            S_list.append(S.detach().clone().cpu().contiguous())
            if (max_batches is not None) and (i + 1 >= max_batches):
                break
    return S_list

def compute_statewise_loss_maps_from_Slist(S_list, ct, model_S):
    # Same logic as your compute_statewise_loss_maps, but iterating S_list directly
    model_S.eval()
    device = ct.device

    nx, ny = int(ct.nx), int(ct.ny)
    types_x = torch.as_tensor(ct.types_x, dtype=torch.float32)
    types_y = torch.as_tensor(ct.types_y, dtype=torch.float32)

    x_min, x_max = types_x[0].item(), types_x[-1].item()
    y_min, y_max = types_y[0].item(), types_y[-1].item()
    sx = (nx - 1) / (x_max - x_min) if nx > 1 else 0.0
    sy = (ny - 1) / (y_max - y_min) if ny > 1 else 0.0

    sum_all = torch.zeros((nx, ny), dtype=torch.float64)
    cnt_all = torch.zeros((nx, ny), dtype=torch.int64)
    sum_state = torch.zeros((3, nx, ny), dtype=torch.float64)
    cnt_state = torch.zeros((3, nx, ny), dtype=torch.int64)

    with torch.enable_grad():
        for S_cpu in S_list:
            S = S_cpu.to(device, non_blocking=True)
            res, _, _ = ct.S_pde_oper(model_S, S)
            res2 = (res.view(-1) ** 2).detach().cpu()

            x_cpu = S_cpu[:, 0]
            y_cpu = S_cpu[:, 1]
            z_cpu = torch.round(S_cpu[:, 2]).to(torch.long)

            ix = torch.clamp(torch.round((x_cpu - x_min) * sx).long(), 0, nx - 1)
            iy = torch.clamp(torch.round((y_cpu - y_min) * sy).long(), 0, ny - 1)

            sum_all.index_put_((ix, iy), res2.to(sum_all.dtype), accumulate=True)
            cnt_all.index_put_((ix, iy), torch.ones_like(ix, dtype=cnt_all.dtype), accumulate=True)

            for s in (0, 1, 2):
                m = (z_cpu == s)
                if m.any():
                    sum_state[s].index_put_((ix[m], iy[m]), res2[m].to(sum_state.dtype), accumulate=True)
                    cnt_state[s].index_put_((ix[m], iy[m]), torch.ones_like(ix[m], dtype=cnt_state.dtype), accumulate=True)

    E_all = sum_all / torch.clamp(cnt_all.to(sum_all.dtype), min=1.0); E_all[cnt_all == 0] = float("nan")
    E_L   = sum_state[0] / torch.clamp(cnt_state[0].to(sum_state.dtype), min=1.0); E_L[cnt_state[0] == 0] = float("nan")
    E_H   = sum_state[1] / torch.clamp(cnt_state[1].to(sum_state.dtype), min=1.0); E_H[cnt_state[1] == 0] = float("nan")
    E_D   = sum_state[2] / torch.clamp(cnt_state[2].to(sum_state.dtype), min=1.0); E_D[cnt_state[2] == 0] = float("nan")

    extent = [x_min, x_max, y_min, y_max]
    return {"E_all": E_all.numpy(), "E_L": E_L.numpy(), "E_H": E_H.numpy(), "E_D": E_D.numpy(), "extent": extent}





