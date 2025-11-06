# -*- coding: utf-8 -*-
"""
Refactor: dQ/dV using common parsing + selection helpers from `arbin_res_utils.py`

Changes vs original:
- Replaces local CSV loader and step/cycle helpers with imports from arbin_res_utils:
    load_frame, build_step_map, select_capacity_column, maybe_flip_voltage, capacity_mAhg
- Keeps dQ/dV math (binning, PCHIP derivative, plotting) inside this script.

Source: original dqdv_for_Arbinres_to_csv.py  # filecite
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colormaps
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from palettable.cartocolors import diverging


plt.rcParams['font.family'] = 'Arial'
# plt.register_cmap("fall", diverging.Fall_4.mpl_colormap)
# plt.register_cmap("geyser", diverging.Geyser_7.mpl_colormap)
# plt.register_cmap("earth", diverging.Earth_4.mpl_colormap)
# plt.register_cmap("armyrose", diverging.ArmyRose_4.mpl_colormap)

# ---------- Import shared helpers ----------
from arbin_res_utils import (
    load_frame,
    build_step_map,
    select_capacity_column,
    maybe_flip_voltage,
    capacity_mAhg,
)
from differential_analysis.SingleICA_FOI import cycle_foi_row, vpeak_shift_between_cycles
from plot_utils import my_cmaps
#my_cmaps.register_ltodefaults()
my_cmaps.register_lfpdefaults()

# ---------------------- Import and Export files ---------------------- #
# Import
csv_path  = r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\LFP - nocccv\csv from res\LFP38.csv"

# Export
foi_outdir         = r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\LFP FOI"
foi_basename       = "FOI_LFP38_15m"
cellname = "LFP 15m"
# Export details: Onset statistics over a cycle range (mean ± SD)
onset_stats_cycles = (5, 43)        # can be (lo, hi) or a list [c5, c6, ...]
compare_cycles     = (1,43)       # for Vmax shifts across two cycles

# ---------------------- USER SETTINGS ---------------------- #
# Electrode mass in grams (update to your value)
mass_g = 0.0059256#0.00531

# Cycle selection
cycles_for_dqdv = [1,5,11,25, 27]# First run test
# cycles_for_dqdv = (1,50) # inclusive range or provide a list instead
discrete_cycles = True      # True if exact cycles in list; False use inclusive range

# Step selection mode per cycle:
#   "minmax" -> discharge step = min Step_Index in cycle; charge step = max Step_Index in cycle
#   "auto"   -> discharge step = Step_Index with largest increase in Discharge_Capacity;
#               charge step    -> Step_Index with largest increase in Charge_Capacity.
step_mode = "auto"

# dQ/dV computation settings
bin_mV         = 2         # voltage bin width (mV) to preserve narrow plateau physics
eval_step_mV   = None      # evaluation grid step (mV); if None, use bin_mV
sigma_bins     = 0.6   #smoothing for Q (Gaussian); if None or 0, no smoothing
sigma_bins2    = 0     # smoothing for Q (Savitzky–Golay); if None or 0, no smoothing
both_positive  = False     # if True, flip discharge sign so both peaks are positive
# voltage_window = [1.72, 1.9]  # Ti2O peak
voltage_window = [3.3, 3.6] #manu
#voltage_window = [2.5, 3.65]  # full


flipped_polarity =True
keep_cap_unflipped = False  # if polarity flipped and you want to keep capacity columns as in raw data. Toggle as needed, usually keep False on default.

# Plot & scaling
cmap_name = "winter"
#cmap_name = "lto_unex" #unex
#cmap_name = "lto_15m" #15min
#cmap_name = "lto_1w" #1week

# cmap_name = "lfp_unex" #unex
# cmap_name = "lfp_unex2" #unex
#cmap_name = "lfp_15m" #15min
#cmap_name = "lfp_1w" #1week
#cmap_name = "armyrose" #15min
use_robust_ylim = True           # percentile-based y-limits
ylim_quantiles  = (0, 100)       # (low%, high%) percentiles for robust ylim


# ---------------------- FOI settings ---------------------- #
# Pre-front windows FOR NOISE (Method A); user-defined here (not in module):
prefront_discharge = (2.6, 2.7)   # V
prefront_charge    = (3.2, 3.25)   # V

# Thresholding parameters for onset
onset_k_sigma      = 5.0
onset_abs_floor    = 400          # mAh/(V·g); ensure >> baseline noise
onset_min_consec   = 3              # consecutive rising bins after crossing

# Peak finding parameters
peak_search_window = (3.35,3.55)   # tighten if needed; None → use all
peak_prominence    = None           # or e.g., 3000 for very noisy data

# FOI computation smoothing (separate from plotting):
foi_sigma_bins     = 0.0            # Gaussian σ (in bins) on Q(V) BEFORE derivative. set to 0.6-1 for FWHM smoothing
foi_sigma_bins2    = 0.0            # optional SG pass on Q(V); usually 0



# ---------------------- dQ/dV utilities (kept local) ---------------------- #
def bin_by_voltage(V: np.ndarray, Q: np.ndarray, bin_mV: float = 0.1):
    """
    Bin (V,Q) by fixed-width voltage bins of size bin_mV (mV).
    Returns representative arrays (Vb, Qb) using the mean within each bin.
    """
    V = np.asarray(V, float); Q = np.asarray(Q, float)
    m = np.isfinite(V) & np.isfinite(Q)
    V, Q = V[m], Q[m]
    if V.size < 4:
        return np.array([]), np.array([])

    step = float(bin_mV) / 1000.0  # mV to V
    vb = np.round(V / step).astype(np.int64)
    order = np.argsort(vb)
    vb, Q = vb[order], Q[order]

    keys, start = np.unique(vb, return_index=True)
    end = np.r_[start[1:], vb.size]
    Qb = np.array([Q[s:e].mean() for s, e in zip(start, end)], float)
    Vb = keys.astype(float) * step
    return Vb, Qb


def dqdv_pchip_binned(V: np.ndarray, Q_mAhg: np.ndarray, bin_mV: float = 0.1,
                      eval_step_mV: float | None = None, sigma_bins: float = 0.0, sigma_bins2: float = 0.0,):
    """
    1) Bin to tiny, fixed ΔV; 2)Optional smoothing of Q(V) 3) PCHIP monotone interpolation; 
    4) analytic derivative evaluated on a uniform grid
    """
     # 1) Bin to fixed ΔV
    Vb, Qb = bin_by_voltage(V, Q_mAhg, bin_mV=bin_mV)
    if Vb.size < 4:
        return np.array([]), np.array([])
    Vb_s = Vb

    if sigma_bins and sigma_bins > 0:
        Qb_smooth1 = gaussian_filter1d(Qb, sigma=float(sigma_bins), mode="reflect", truncate=float(2.0))
    else:
        Qb_smooth1 = Qb
    
    if sigma_bins2 and sigma_bins2 > 0:
        n = Qb_smooth1.size
        # window length chosen minimally, preserving your single-arg API
        if sigma_bins2 < 1.0:
            # MATLAB 'smooth' fraction-of-data: span ~= percent * n, make it odd
            w = int(round(max(5, sigma_bins2 * n)))
        else:
            # interpret sigma_bins as a bins-scale: window ≈ 2*sigma + 1
            w = int(round(2 * float(sigma_bins2) + 1))

        # enforce odd window and valid bounds
        if w % 2 == 0:
            w += 1
        w = max(3, min(w, n - (1 - n % 2)))  # ensure w <= n and odd

        # choose a small polyorder; constrain to < w
        poly = 2 if w < 7 else 3
        poly = min(poly, w-1)

        Qb_smooth = savgol_filter(Qb_smooth1, window_length=w, polyorder=poly, mode='interp')
    else:
        Qb_smooth = Qb_smooth1

    p = PchipInterpolator(Vb_s, Qb_smooth, extrapolate=False)
    step = float((eval_step_mV if eval_step_mV is not None else bin_mV)) / 1000.0  # V
    Vg = np.arange(Vb_s.min(), Vb_s.max() + 0.5 * step, step)
    
    dQdV = p.derivative(1)(Vg)
    #dQdV=gaussian_filter1d(dQdV, sigma=1, mode="reflect", truncate=float(2.0)) #set to 0 by default for highest fidelity, 0.6-1 for FWHM smoothing
    m = np.isfinite(dQdV)

    return Vg[m], dQdV[m]


# ---------------------- Plotting ---------------------- #
def plot_dqdv_by_step(df: pd.DataFrame, mass_g: float, cycles=None, mode="minmax",
                      bin_mV=None, eval_step_mV=None, sigma_bins=None, sigma_bins2=None, 
                      both_positive=False, voltage_window=None,
                      cmap_name="viridis", 
                      use_robust_ylim=True, ylim_quantiles=(0.5, 99.5),cellname=cellname):
    import matplotlib as mpl

    mapping = build_step_map(df, cycles=cycles, mode=mode)
    if not mapping:
        print("No cycles/steps found for dQ/dV plotting.")
        return

    cycles_sorted = sorted(mapping.keys())
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name, len(cycles_sorted))

    all_vals = []  # for robust ylim

    for i, c in enumerate(cycles_sorted):
        dis_step, chg_step = mapping[c]
        cdf = df[df["Cycle_Index"] == c]
        # If wiring is flipped, the physical discharge occurs during the program "charge" step
        if flipped_polarity and not keep_cap_unflipped:
            dis_step, chg_step = chg_step, dis_step

        # Thicker line for the first cycle for readability
        lw = 3# if i == 0 else 1.5
        ls = "-"
        zorder = 10 if i == 0 else 3

        # --- Discharge ---
        dis = cdf[cdf["Step_Index"] == dis_step]
        if not dis.empty:
            qcol_dis = select_capacity_column(is_discharge=True, flipped=flipped_polarity, keep_cap=keep_cap_unflipped)
            if qcol_dis in dis.columns and "Voltage" in dis.columns:
                V = maybe_flip_voltage(pd.to_numeric(dis["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
                Q = capacity_mAhg(pd.to_numeric(dis[qcol_dis], errors="coerce").to_numpy(), mass_g=mass_g)

                Vd, dQdVd = dqdv_pchip_binned(V, Q, bin_mV=bin_mV,
                                               eval_step_mV=eval_step_mV, sigma_bins=sigma_bins, sigma_bins2=sigma_bins2)
                if Vd.size:
                    if voltage_window is not None and len(voltage_window) == 2:
                        vmin, vmax = map(float, voltage_window)
                        m = (Vd >= vmin) & (Vd <= vmax)
                        Vd, dQdVd = Vd[m], dQdVd[m]
                    if both_positive:
                        dQdVd = -dQdVd
                    all_vals.append(dQdVd)
                    ax.plot(Vd, dQdVd, lw=lw, ls=ls, zorder=zorder,
                             #label="discharge" if i == 0 else None, 
                             color=cmap(i))

        # --- Charge ---
        chg = cdf[cdf["Step_Index"] == chg_step]
        if not chg.empty:
            qcol_chg = select_capacity_column(is_discharge=False, flipped=flipped_polarity, keep_cap=keep_cap_unflipped)
            if qcol_chg in chg.columns and "Voltage" in chg.columns:
                V = maybe_flip_voltage(pd.to_numeric(chg["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
                Q = capacity_mAhg(pd.to_numeric(chg[qcol_chg], errors="coerce").to_numpy(), mass_g=mass_g)

                Vc, dQdVc = dqdv_pchip_binned(V, Q, bin_mV=bin_mV,
                                               eval_step_mV=eval_step_mV, sigma_bins=sigma_bins, sigma_bins2=sigma_bins2)
                if Vc.size:
                    if voltage_window is not None and len(voltage_window) == 2:
                        vmin, vmax = map(float, voltage_window)
                        m = (Vc >= vmin) & (Vc <= vmax)
                        Vc, dQdVc = Vc[m], dQdVc[m]
                    all_vals.append(dQdVc)
                    ax.plot(Vc, dQdVc, lw=lw, ls=ls, zorder=zorder,
                             #label="charge" if i == 0 else None, 
                             color=cmap(i))
        
        # Add a dummy line (invisible) just for legend entry
        ax.plot([], [], color=cmap(i), lw=2, label=f"Cycle {c}")

    ax.set_xlabel("Voltage (V)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r'dQ/dV ($\frac{mAh}{V \cdot g}$)', fontsize=18, fontweight='bold')
    title = "dQ/dV – " + str(cellname)
    # ax.set_title(title, fontsize=18, fontweight='bold')

    if both_positive:
        title += " [discharge flipped positive]"
    ax.set_title(title, fontsize=18, fontweight='bold')

    if discrete_cycles == False:
        # Continuous: colorbar keyed by cycle index
        norm = mpl.colors.Normalize(vmin=min(cycles_sorted), vmax=max(cycles_sorted))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Cycle Index", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
    else:
        # Discrete: legend with labels
        ax.legend(fontsize=15, loc='lower right', shadow=True,)
        

    # Robust y-limits to keep features visible
    if use_robust_ylim and all_vals:
        vals = np.concatenate(all_vals)
        lo = np.nanpercentile(vals, ylim_quantiles[0])
        hi = np.nanpercentile(vals, ylim_quantiles[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.set_ylim(lo, hi)

    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', labelsize=14)
    plt.show()

def compute_foi_table_and_save(df: pd.DataFrame,
                               mass_g: float,
                               cycles=None,
                               mode="auto",
                               bin_mV: float = 2.0,
                               eval_step_mV: float | None = None,
                               foi_sigma_bins: float = 0.0,
                               foi_sigma_bins2: float = 0.0,
                               voltage_window=None,
                               both_positive: bool = False,
                               # FOI params from user settings:
                               prefront_discharge: tuple = prefront_discharge,
                               prefront_charge: tuple = prefront_charge,
                               onset_k_sigma: float = onset_k_sigma,
                               onset_abs_floor: float = onset_abs_floor,
                               onset_min_consec: int = onset_min_consec,
                               peak_search_window: tuple | None = peak_search_window,
                               peak_prominence: float | None = peak_prominence,
                               outdir: str = foi_outdir,
                               basename: str = foi_basename,
                               onset_stats_cycles=onset_stats_cycles,
                               compare_cycles: tuple = compare_cycles):
    """
    Compute FOIs for each selected cycle (charge & discharge), print a compact table,
    and save CSV (+ optional TXT summary).
    """
    os.makedirs(outdir or ".", exist_ok=True)

    # 1) Build step map once
    mapping = build_step_map(df, cycles=cycles, mode=mode)
    if not mapping:
        print("No cycles/steps found for FOI computation.")
        return None

    rows = {}
    for c in sorted(mapping.keys()):
        dis_step, chg_step = mapping[c]
        cyc_df = df[df["Cycle_Index"] == c]

        # Map physical discharge/charge if polarity flipped
        if flipped_polarity:
            dis_step, chg_step = chg_step, dis_step

        # --- Discharge data ---
        dis = cyc_df[cyc_df["Step_Index"] == dis_step]
        # --- Charge data ---
        chg = cyc_df[cyc_df["Step_Index"] == chg_step]

        # Extract V, Q, then compute dQ/dV for each branch using your existing pipeline
        if not dis.empty:
            qcol_dis = select_capacity_column(is_discharge=True,  flipped=flipped_polarity, keep_cap=keep_cap_unflipped)
            Vd_raw   = maybe_flip_voltage(pd.to_numeric(dis["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
            Qd_raw   = capacity_mAhg(pd.to_numeric(dis[qcol_dis], errors="coerce").to_numpy(), mass_g=mass_g)
            Vd, dQdVd = dqdv_pchip_binned(Vd_raw, Qd_raw, bin_mV=bin_mV,
                                          eval_step_mV=eval_step_mV,
                                          sigma_bins=foi_sigma_bins, sigma_bins2=foi_sigma_bins2)
        else:
            Vd, dQdVd = np.array([]), np.array([])

        if not chg.empty:
            qcol_chg = select_capacity_column(is_discharge=False, flipped=flipped_polarity, keep_cap=keep_cap_unflipped)
            Vc_raw   = maybe_flip_voltage(pd.to_numeric(chg["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
            Qc_raw   = capacity_mAhg(pd.to_numeric(chg[qcol_chg], errors="coerce").to_numpy(), mass_g=mass_g)
            Vc, dQdVc = dqdv_pchip_binned(Vc_raw, Qc_raw, bin_mV=bin_mV,
                                          eval_step_mV=eval_step_mV,
                                          sigma_bins=foi_sigma_bins, sigma_bins2=foi_sigma_bins2)
        else:
            Vc, dQdVc = np.array([]), np.array([])

        # Clip by voltage window if requested
        def _clip(V, Y):
            if V.size == 0: return V, Y
            if voltage_window is None: return V, Y
            vmin, vmax = map(float, voltage_window)
            m = (V >= vmin) & (V <= vmax)
            return V[m], Y[m]

        Vd, dQdVd = _clip(Vd, dQdVd)
        Vc, dQdVc = _clip(Vc, dQdVc)

        # Compute FOIs for this cycle
        row = cycle_foi_row(
            Vd, dQdVd, Vc, dQdVc,
            prewin_dis=prefront_discharge, prewin_chg=prefront_charge,
            k_sigma=onset_k_sigma, abs_floor=onset_abs_floor, min_consecutive=onset_min_consec,
            both_positive=both_positive, peak_window=peak_search_window,
            peak_prominence=peak_prominence, cycle_index=c
        )
        rows[int(c)] = row

    # 2) To DataFrame
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # Pretty print
    display_cols = [
        "Vonset_dis", "Vonset_chg",
        "Vmax_dis", "Vmax_chg",
        "Hmax_dis_abs", "Hmax_chg_abs",
        "FWHM_dis", "FWHM_chg",
        "delta_max_dis", "delta_max_chg",
        "Vmax_chg_minus_dis"
    ]
    print("\nFOIs (per cycle):")
    print(table[display_cols].to_string(float_format=lambda v: f"{v:8.4f}"))

    # 3) Vpeak shifts between two cycles
    c1, c2 = int(compare_cycles[0]), int(compare_cycles[1])
    shifts = vpeak_shift_between_cycles(rows, c1, c2)
    print(f"\nΔV_peak,max between cycles {c1} and {c2}:")
    for k, v in shifts.items():
        print(f"  {k}: {v:8.4f} V")

    # 4) Onset statistics over user-selected cycle range
    def _sel_cycles(cyc_sel):
        if isinstance(cyc_sel, tuple) and len(cyc_sel) == 2:
            lo, hi = cyc_sel
            return [c for c in table.index if int(lo) <= int(c) <= int(hi)]
        else:
            return [int(x) for x in np.atleast_1d(cyc_sel)]

    onset_sel = _sel_cycles(onset_stats_cycles)
    if onset_sel:
        sub = table.loc[onset_sel]
        mu_dis = float(np.nanmean(sub["Vonset_dis"].values))
        sd_dis = float(np.nanstd(sub["Vonset_dis"].values, ddof=1)) if len(sub) > 1 else 0.0
        mu_chg = float(np.nanmean(sub["Vonset_chg"].values))
        sd_chg = float(np.nanstd(sub["Vonset_chg"].values, ddof=1)) if len(sub) > 1 else 0.0
        print(f"\nOnset statistics over cycles {onset_sel}:")
        print(f"  Discharge onset   = {mu_dis:.4f} ± {sd_dis:.4f} V")
        print(f"  Charge onset      = {mu_chg:.4f} ± {sd_chg:.4f} V")

    # 5) Save CSV (+ optional TXT)
    csv_path = os.path.join(outdir, f"{basename}_foi.csv")
    table.to_csv(csv_path, index_label="cycle")
    print(f"\n[Saved] FOIs → {csv_path}")

    # Also save a tiny TXT summary with shifts + onset stats (optional)
    txt_path = os.path.join(outdir, f"{basename}_summary.txt")
    with open(txt_path, "w",encoding="utf-8") as f:
        f.write("FOI summary\n===========\n\n")
        f.write(f"Cycles analysed: {sorted(rows.keys())}\n\n")
        f.write(f"ΔV_peak,max between cycles {c1} and {c2}:\n")
        for k, v in shifts.items():
            f.write(f"  {k}: {v:.6f} V\n")
        if onset_sel:
            f.write(f"\nOnset statistics over cycles {onset_sel}:\n")
            f.write(f"  Discharge onset   = {mu_dis:.6f} ± {sd_dis:.6f} V\n")
            f.write(f"  Charge onset      = {mu_chg:.6f} ± {sd_chg:.6f} V\n")
    print(f"[Saved] Summary → {txt_path}")

    return table



def main():
    df = load_frame(csv_path)  # from arbin_res_utils
    plot_dqdv_by_step(
        df,
        mass_g=mass_g,
        cycles=cycles_for_dqdv, 
        mode=step_mode,
        bin_mV=bin_mV,
        eval_step_mV=eval_step_mV,
        sigma_bins=sigma_bins,
        sigma_bins2=sigma_bins2,
        both_positive=both_positive,
        voltage_window=voltage_window,
        cmap_name=cmap_name,
        use_robust_ylim=use_robust_ylim,
        ylim_quantiles=ylim_quantiles, cellname=cellname
    )
    # #FOIs (computed with their own smoothing knobs; defaults keep onset from σ=0)
    _ = compute_foi_table_and_save(
        df, mass_g=mass_g,
        cycles=cycles_for_dqdv, mode=step_mode,
        bin_mV=bin_mV, eval_step_mV=eval_step_mV,
        foi_sigma_bins=foi_sigma_bins, foi_sigma_bins2=foi_sigma_bins2,
        voltage_window=voltage_window, both_positive=both_positive,
        # FOI params come from the block at the top of this file
    )


if __name__ == "__main__":
    main()
