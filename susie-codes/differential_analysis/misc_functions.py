import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from palettable.cartocolors import diverging

# ---------- Import shared helpers ----------
from arbin_res_utils import (
    load_frame,
    build_step_map,
    select_capacity_column,
    maybe_flip_voltage,
    capacity_mAhg,
)

# ---------------------- USER SETTINGS ---------------------- #
csv_path  = r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\CSV from RES\LTOc288.csv"

# Electrode mass in grams (update to your value)
mass_g = 0.0039996  # 7.02 mg

# Cycle selection
cycles_for_dqdv = [1,5,25,38,50] # inclusive range or provide a list instead
discrete_cycles = False      # True if exact cycles in list; False use inclusive range

# Step selection mode per cycle:
#   "minmax" -> discharge step = min Step_Index in cycle; charge step = max Step_Index in cycle
#   "auto"   -> discharge step = Step_Index with largest increase in Discharge_Capacity;
#               charge step    -> Step_Index with largest increase in Charge_Capacity.
step_mode = "auto"

# dQ/dV computation settings
bin_mV         = 2         # voltage bin width (mV) to preserve narrow plateau physics
eval_step_mV   = None      # evaluation grid step (mV); if None, use bin_mV
sigma_bins     = None    #smoothing for V (Savitzky–Golay); if None or 0, no smoothing
sigma_bins2    = None     # smoothing for Q (Savitzky–Golay); if None or 0, no smoothing
both_positive  = False     # if True, flip discharge sign so both peaks are positive
voltage_window = [1.45, 1.7]  # set to None to disable clipping

# iR correction settings
delta_corrections_mV = [15.6, 13.2, 9.2] #get this from running iR function in single_plotdata.py, set as None to disable iR correction
delta_corrections = [v / 1000.0 for v in delta_corrections_mV] if delta_corrections_mV else None #turn above mV values into V


cellname = "LTOc288: 15 mins"
flipped_polarity = False

# Plot & scaling
cmap_name = "winter"
use_robust_ylim = True           # percentile-based y-limits
ylim_quantiles  = (0, 100)       # (low%, high%) percentiles for robust ylim
# ----------------------------------------------------------- #

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
    1) Bin to tiny, fixed ΔV; 2) PCHIP monotone interpolation; 3) analytic derivative
    evaluated on a uniform grid; 4) optional Gaussian smoothing of dQ/dV.
    """
     # 1) Bin to fixed ΔV
    Vb, Qb = bin_by_voltage(V, Q_mAhg, bin_mV=bin_mV)
    if Vb.size < 4:
        return np.array([]), np.array([])

    # 2) OPTIONAL: Savitzky–Golay smoothing of *Vb* (after binning)
    if sigma_bins and sigma_bins > 0:
        n = Vb.size
        if sigma_bins < 1.0:
            w = int(np.ceil(sigma_bins * n))     # % of binned points
        else:
            w = int(round(2 * float(sigma_bins) + 1))  # bins-scale → window
        w = max(3, w | 1)                        # odd, ≥3
        if w >= n:
            w = n - (1 - n % 2)
        poly = 2 if w < 7 else 3

        Vb_s = savgol_filter(Vb, window_length=w, polyorder=poly, mode='interp')

        # Ensure strictly increasing x for PCHIP (very important)
        eps = 1e-9
        Vb_s = np.maximum.accumulate(Vb_s + eps * np.arange(n))
    else:
        Vb_s = Vb
    Vb_s = Vb

    if sigma_bins and sigma_bins > 0:
        Qb_smooth = gaussian_filter1d(Qb, sigma=float(sigma_bins), mode="reflect", truncate=float(2.0))
    else:
        Qb_smooth = Qb
    
    if sigma_bins2 and sigma_bins2 > 0:
        n = Qb.size
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

        Qb_smooth = savgol_filter(Qb, window_length=w, polyorder=poly, mode='interp')
    else:
        Qb_smooth = Qb

    p = PchipInterpolator(Vb_s, Qb_smooth, extrapolate=False)
    step = float((eval_step_mV if eval_step_mV is not None else bin_mV)) / 1000.0  # V
    Vg = np.arange(Vb_s.min(), Vb_s.max() + 0.5 * step, step)
    
    dQdV = p.derivative(1)(Vg)

    # if sigma_bins2 and sigma_bins2 > 0:
    #     dQdV = gaussian_filter1d(dQdV, sigma=float(sigma_bins2)
    m = np.isfinite(dQdV)

    return Vg[m], dQdV[m]

# ---------------------- Plotting ICA with iR adjustment ---------------------- #
def plot_dqdv_ir_corrected_by_step(df: pd.DataFrame, mass_g: float,
                                   cycles=None,                      # e.g., [1, 2, 50]
                                   deltas= None,                      # list, dict {cycle: delta}, float, or None
                                   mode="auto",
                                   bin_mV=None, eval_step_mV=None,
                                   sigma_bins=None, sigma_bins2=None, 
                                   both_positive=False,
                                   voltage_window=None,
                                   cmap_name="Spectral",
                                   use_robust_ylim=True,
                                   ylim_quantiles=(0, 100)):
    """
    Like plot_dqdv_by_step, but apply a user-provided constant delta per cycle:
        charge: V' = V - delta
        discharge: V' = V + delta
    If deltas is None, plots raw dQ/dV.
    """
    import matplotlib as mpl

    mapping = build_step_map(df, cycles=cycles, mode=mode)
    if not mapping:
        print("No cycles/steps found for dQ/dV plotting.")
        return

    # Normalize cycles + deltas
    cycles_sorted = sorted(mapping.keys())

    delta_map = {}
    if deltas is None:
        delta_map = {c: 0.0 for c in cycles_sorted}
    elif isinstance(deltas, dict):
        # Use 0 for cycles not in dict
        delta_map = {c: float(deltas.get(c, 0.0)) for c in cycles_sorted}
    elif np.isscalar(deltas):
        delta_map = {c: float(deltas) for c in cycles_sorted}
    else:
        # assume list/tuple in same order as `cycles`
        if len(deltas) != len(cycles_sorted):
            raise ValueError("Length of `deltas` must match number of cycles or provide a dict {cycle: delta}.")
        delta_map = {c: float(d) for c, d in zip(cycles_sorted, deltas)}

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name, len(cycles_sorted))
    all_vals = []

    for i, c in enumerate(cycles_sorted):
        dis_step, chg_step = mapping[c]
        cdf = df[df["Cycle_Index"] == c]

        # flip step meaning if polarity flipped
        local_dis_step, local_chg_step = (chg_step, dis_step) if flipped_polarity else (dis_step, chg_step)
        delta = float(delta_map.get(c, 0.0))

        # Thicker line for first cycle
        lw = 4 if i == 0 else 1.5
        ls = "-"
        zorder = 10 if i == 0 else 1
        color = cmap(i)

        # ---- Discharge ----
        dis = cdf[cdf["Step_Index"] == local_dis_step]
        if not dis.empty:
            qcol_dis = select_capacity_column(is_discharge=True,  flipped=flipped_polarity)
            if qcol_dis in dis.columns and "Voltage" in dis.columns:
                V = maybe_flip_voltage(pd.to_numeric(dis["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
                V = V + delta  # iR correction for discharge
                Q = capacity_mAhg(pd.to_numeric(dis[qcol_dis], errors="coerce").to_numpy(), mass_g=mass_g)

                Vd, dQdVd = dqdv_pchip_binned(V, Q, bin_mV=bin_mV, eval_step_mV=eval_step_mV, sigma_bins=sigma_bins, sigma_bins2=sigma_bins2)
                if Vd.size:
                    if voltage_window is not None and len(voltage_window) == 2:
                        vmin, vmax = map(float, voltage_window)
                        m = (Vd >= vmin) & (Vd <= vmax)
                        Vd, dQdVd = Vd[m], dQdVd[m]
                    if both_positive:
                        dQdVd = -dQdVd
                    all_vals.append(dQdVd)
                    ax.plot(Vd, dQdVd, lw=lw, ls=ls, zorder=zorder, color=color)

        # ---- Charge ----
        chg = cdf[cdf["Step_Index"] == local_chg_step]
        if not chg.empty:
            qcol_chg = select_capacity_column(is_discharge=False, flipped=flipped_polarity)
            if qcol_chg in chg.columns and "Voltage" in chg.columns:
                V = maybe_flip_voltage(pd.to_numeric(chg["Voltage"], errors="coerce").to_numpy(), flipped=flipped_polarity)
                V = V - delta  # iR correction for charge
                Q = capacity_mAhg(pd.to_numeric(chg[qcol_chg], errors="coerce").to_numpy(), mass_g=mass_g)

                Vc, dQdVc = dqdv_pchip_binned(V, Q, bin_mV=bin_mV, eval_step_mV=eval_step_mV, sigma_bins=sigma_bins, sigma_bins2=sigma_bins2)
                if Vc.size:
                    if voltage_window is not None and len(voltage_window) == 2:
                        vmin, vmax = map(float, voltage_window)
                        m = (Vc >= vmin) & (Vc <= vmax)
                        Vc, dQdVc = Vc[m], dQdVc[m]
                    all_vals.append(dQdVc)
                    ax.plot(Vc, dQdVc, lw=lw, ls=ls, zorder=zorder, color=color)

        # Add a dummy line (invisible) just for legend entry
        ax.plot([], [], color=cmap(i), lw=2, label=f"Cycle {c}")

    ax.set_xlabel("Voltage (V)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r'dQ/dV ($\frac{mAh}{V \cdot g}$)', fontsize=18, fontweight='bold')
    title = "dQ/dV – iR-corrected" if any(abs(v) > 0 for v in delta_map.values()) else "dQ/dV – raw"
    ax.set_title(title + f" – {cellname}", fontsize=16, fontweight='bold')

    #norm = mpl.colors.Normalize(vmin=min(cycles_sorted), vmax=max(cycles_sorted))
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    #cbar = fig.colorbar(sm, ax=ax)
    #cbar.set_label("Cycle Index", fontsize=14, fontweight='bold')
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
        ax.legend(title="Cycle Index", fontsize=12)

    if use_robust_ylim and all_vals:
        vals = np.concatenate(all_vals)
        lo = np.nanpercentile(vals, ylim_quantiles[0]); hi = np.nanpercentile(vals, ylim_quantiles[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.set_ylim(lo, hi)

    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', labelsize=14)
    plt.show()

def main():
    df = load_frame(csv_path)  # from arbin_res_utils

    plot_dqdv_ir_corrected_by_step(
         df,
         mass_g=mass_g,
         cycles=cycles_for_dqdv,
         deltas=delta_corrections,          # or a dict {cycle: delta}, or a single float, or None for raw
         mode=step_mode,
         bin_mV=bin_mV,
         eval_step_mV=eval_step_mV,
         sigma_bins=sigma_bins, sigma_bins2=sigma_bins2,
         both_positive=both_positive,
         voltage_window=voltage_window,
         cmap_name=cmap_name,
         use_robust_ylim=use_robust_ylim,
         ylim_quantiles=ylim_quantiles,)


if __name__ == "__main__":
    main()