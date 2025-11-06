# -*- coding: utf-8 -*-
"""
Refactored (mixed CC-only & CC–CV): CE & voltage profiles using battery_utils.py

Features:
- Per-cycle toggle: some cycles use CC–CV (phase), others use CC-only (step).
- CE can be computed cycle-wide (robust for CC–CV and OK for CC) or via steps for CC-only cycles.
- Voltage profiles follow the same per-cycle toggle for plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from arbin_res_load import parse_arbinres 
from palettable.cartocolors import diverging 
# from palettable.cmocean import diverging 

plt.rcParams['font.family'] = 'Arial'
""" plt.register_cmap("fall", diverging.Fall_4.mpl_colormap)
plt.register_cmap("geyser", diverging.Geyser_4.mpl_colormap)
plt.register_cmap("earth", diverging.Earth_4.mpl_colormap)
plt.register_cmap("armyrose", diverging.ArmyRose_4.mpl_colormap) """
#plotcolor = "earth" #unex
#plotcolor = "geyser" #15min
#plotcolor = "fall" #1week
#plotcolor = "armyrose" #15min

# plt.register_cmap("fall", diverging.Delta_4.mpl_colormap)
# plt.register_cmap("armyrose", diverging.ArmyRose_4.mpl_colormap)
# plt.register_cmap("earth", diverging.Earth_4.mpl_colormap)
#plotcolor = "earth" #unex
#plotcolor = "geyser" #15min
#plotcolor = "fall" #1week

from arbin_res_utils import (
    ensure_csv_from_res, load_frame,
    cycles_list, subset_by_cycles, sort_by_time,
    select_capacity_column, maybe_flip_voltage, capacity_mAhg,
    build_step_map, get_step_frames,
    build_step_map_manual,          
    capacity_column_for_step,       
    iter_phases, phase_frames_by_cycle,
    iter_frames_mixed, get_cycle_frames_mixed,
    cycles_list
)

from differential_analysis.SingleICA_FOI import cycle_foi_row, vpeak_shift_between_cycles
from plot_utils import my_cmaps
# my_cmaps.register_ltodefaults()
my_cmaps.register_lfpdefaults()

# ---------------------- USER SETTINGS ---------------------- #
res_path  = r""  # leave empty to skip RES->CSV
csv_path  = r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\LFP - nocccv\csv from res\LFP11.csv"

cellname = "LTOc277 1 week"
flipped_polarity = True
keep_cap_unflipped = True  # if polarity flipped and you want to keep capacity columns as in raw data. Toggle as needed, usually keep False on default.

mass_g =0.0041346 # active mass in grams

#color:
#plotcolor = "lto_unex" #unex
#plotcolor = "lto_15m" #15min
#plotcolor = "lto_1w" #1week

# plotcolor = "lfp_unex" #unex
#plotcolor = "lfp_unex2" #unex
#plotcolor = "lfp_15m" #15min
# plotcolor = "lfp_15m2" #15min2
#plotcolor = "lfp_1w" #1week

plotcolor = "viridis"

# Cycle selections
cycles_for_ce       = (1, 50)
cycles_for_profiles = (1,50)#[1,5, 49]
cycles_for_vt       = (1, 50)
discrete_cycles = False      # True if exact cycles in list; False use inclusive range

# ---- Per-cycle mode toggles ----
# List cycles that should use CC–CV (phase) instead of CC-only (steps).
cccv_cycles = None   # example: only early formation cycles have CV thus [1, 2, 3, 4]; edit as needed

# When a cycle is not in cccv_cycles, CC-only selection uses this step mode:
step_mode = "auto"           # 'auto' or 'minmax'
manual = False  # True to use your manual step map below (uncomment it)
#user_steps = {"default": {"charge": 4, "discharge": 5}} # Step dict

# Phase extraction sensitivity for CC–CV cycles
phase_dQ_min = 1e-8           # try 1e-8 to suppress noise if needed

# ---- CE options ----
# Use cycle-wide capacities for CE (robust for CC–CV and fine for CC-only), or mixed:
ce_use_cycle_wide = True     # True: use whole-cycle capacity for all cycles
ce_ratio = "dis_over_chg"    # 'chg_over_dis' (your previous) or 'dis_over_chg'
ce_aggregate_method = "max"  # 'max' or 'span' for cycle-wide capacity

# Voltage vs time axes (optional)
vt_xlim = None
vt_ylim = None #(-0.2, 0.2)
# ----------------------------------------------------------- #


# ---------------------- CE COMPUTATION (local only) ---------------------- #
def compute_ce_cycle_wide(df: pd.DataFrame,
                          mass_g: float,
                          cycles=None,
                          flipped: bool = False,
                          keep_cap: bool = False,
                          ratio: str = "chg_over_dis",
                          method: str = "max") -> pd.DataFrame:
    """
    CE from whole-cycle capacity (includes CC+CV). Works for all cycles uniformly.
    """
    rows = []
    for cyc in cycles_list(df, cycles):
        cyc_df = df[pd.to_numeric(df["Cycle_Index"], errors="coerce").astype("Int64") == int(cyc)]
        if cyc_df.empty:
            continue

        qdis_col = select_capacity_column(True, flipped, keep_cap)
        qchg_col = select_capacity_column(False, flipped, keep_cap)

        qdis = pd.to_numeric(cyc_df.get(qdis_col, pd.Series(dtype=float)), errors="coerce").dropna()
        qchg = pd.to_numeric(cyc_df.get(qchg_col, pd.Series(dtype=float)), errors="coerce").dropna()
        if qdis.empty or qchg.empty:
            continue

        if method == "span":
            Qdis_Ah = float(qdis.max() - qdis.min())
            Qchg_Ah = float(qchg.max() - qchg.min())
        else:
            Qdis_Ah = float(qdis.max())
            Qchg_Ah = float(qchg.max())

        Dis_mAhg = capacity_mAhg(Qdis_Ah, mass_g)
        Chg_mAhg = capacity_mAhg(Qchg_Ah, mass_g)
        ce = np.nan
        if Dis_mAhg > 0 and Chg_mAhg > 0:
            ce = (Chg_mAhg / Dis_mAhg) * 100.0 if ratio == "chg_over_dis" else (Dis_mAhg / Chg_mAhg) * 100.0

        rows.append({
            "Cycle_Index": int(cyc),
            "Discharge_Capacity_mAhg": Dis_mAhg,
            "Charge_Capacity_mAhg": Chg_mAhg,
            "CE_percent": ce
        })
    return pd.DataFrame(rows).sort_values("Cycle_Index")


def compute_ce_mixed(df: pd.DataFrame,
                     mass_g: float,
                     cycles=None,
                     cccv_cycles=None,
                     step_mode: str = "auto",
                     flipped: bool = False, keep_cap: bool = False,
                     ratio: str = "chg_over_dis",
                     cycle_method: str = "max") -> pd.DataFrame:
    """
    CE per cycle with per-cycle toggle:
      - cycles in cccv_cycles -> use whole-cycle capacity (CC+CV)
      - other cycles          -> use step-selected max capacity (CC-only)
    """
    cccv_set = set(int(x) for x in (cccv_cycles or []))
    rows = []
    for cyc in cycles_list(df, cycles):
        cyc_i = int(cyc)
        if cyc_i in cccv_set:
            # cycle-wide capacity
            ce_df = compute_ce_cycle_wide(df, mass_g, cycles=[cyc_i],
                                          flipped=flipped, ratio=ratio, method=cycle_method)
            if not ce_df.empty:
                rows.append(ce_df.iloc[0].to_dict())
            continue

        # CC-only: use step-selected frames
        dis_df, chg_df = get_step_frames(df, cyc_i, mode=step_mode)
        if dis_df.empty or chg_df.empty:
            continue
        qdis_col = select_capacity_column(True, flipped, keep_cap)
        qchg_col = select_capacity_column(False, flipped, keep_cap)
        Qdis_Ah = pd.to_numeric(dis_df.get(qdis_col, pd.Series(dtype=float)), errors="coerce").max()
        Qchg_Ah = pd.to_numeric(chg_df.get(qchg_col, pd.Series(dtype=float)), errors="coerce").max()
        Dis_mAhg = capacity_mAhg(Qdis_Ah, mass_g)
        Chg_mAhg = capacity_mAhg(Qchg_Ah, mass_g)

        ce = np.nan
        if Dis_mAhg > 0 and Chg_mAhg > 0:
            ce = (Chg_mAhg / Dis_mAhg) * 100.0 if ratio == "chg_over_dis" else (Dis_mAhg / Chg_mAhg) * 100.0

        rows.append({
            "Cycle_Index": cyc_i,
            "Discharge_Capacity_mAhg": Dis_mAhg,
            "Charge_Capacity_mAhg": Chg_mAhg,
            "CE_percent": ce
        })
    return pd.DataFrame(rows).sort_values("Cycle_Index")


# ---------------------- PLOTTING ---------------------- #
def plot_ce(ce_df: pd.DataFrame):
    if ce_df.empty:
        print("No CE data to plot.")
        return
    plt.figure(figsize=(6, 5))
    plt.plot(ce_df["Cycle_Index"], ce_df["CE_percent"], marker="o", lw=1.2)
    plt.xlabel("Cycle Index", fontsize=18, fontweight='bold')
    plt.ylabel("Coulombic Efficiency (%)", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_voltage_profiles_mixed(df: pd.DataFrame,
                                mass_g: float,
                                cycles=None,
                                cccv_cycles=None,
                                step_mode: str = "auto",
                                flipped: bool = False, keep_cap: bool = False,
                                dQ_min: float = 0.0, plotcolor: str = "geyser",
                                title_suffix: str = ""):
    """
    Voltage–capacity profiles per cycle with per-cycle toggle:
      - cycles in cccv_cycles -> CC–CV (phase) selection
      - others               -> CC-only (step) selection
    """
    selected = list(cycles_list(df, cycles))
    if not selected:
        print("No cycles found to plot.")
        return

    cccv_set = set(int(x) for x in (cccv_cycles or []))
    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap(plotcolor, len(selected))

    for i, cyc in enumerate(selected):
        use_phase = (int(cyc) in cccv_set)

        if use_phase:
            dis_df, chg_df = phase_frames_by_cycle(df, cycle=int(cyc), flipped=flipped, dQ_min=dQ_min)
        else:
            dis_df, chg_df = get_step_frames(df, cycle=int(cyc), mode=step_mode)

        # Discharge trace
        if not dis_df.empty:
            qcol_dis = select_capacity_column(True, flipped, keep_cap)
            Qdis_mAhg = capacity_mAhg(pd.to_numeric(dis_df[qcol_dis], errors="coerce").to_numpy(), mass_g=mass_g)
            Vdis = maybe_flip_voltage(pd.to_numeric(dis_df["Voltage"], errors="coerce").to_numpy(), flipped=flipped)
            plt.plot(Qdis_mAhg, Vdis, lw=3, color=cmap(i), label=f"Cycle {cyc}" if discrete_cycles else "_nolegend_")

        # Charge trace
        if not chg_df.empty:
            qcol_chg = select_capacity_column(False, flipped, keep_cap)
            Qchg_mAhg = capacity_mAhg(pd.to_numeric(chg_df[qcol_chg], errors="coerce").to_numpy(), mass_g=mass_g)
            Vchg = maybe_flip_voltage(pd.to_numeric(chg_df["Voltage"], errors="coerce").to_numpy(), flipped=flipped)
            plt.plot(Qchg_mAhg, Vchg, lw=3, color=cmap(i),label="_nolegend_")

    plt.xlabel("Capacity (mAh g$^{-1}$)", fontsize=18, fontweight='bold')
    plt.ylabel("Voltage (V)", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.ylim(2.5, 3.65)
    #norm = mpl.colors.Normalize(vmin=min(selected), vmax=max(selected))
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm)
    #cbar.set_label("Cycle Index", fontsize=14, fontweight='bold')
    #cbar.ax.tick_params(labelsize=12)
    if not discrete_cycles:
    # Continuous: colorbar keyed by cycle index
        norm = mpl.colors.Normalize(vmin=min(selected), vmax=max(selected))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label("Cycle Index", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
    else:
    # Discrete: legend with labels
        plt.legend( fontsize=15, loc='lower center', shadow=True,)

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_voltage_vs_time(df: pd.DataFrame,
                         label: str = None,
                         cycles=None,
                         flipped: bool = False,
                         xlim=None, ylim=None,
                         lw: float = 1.5):
    """Continuous: Voltage vs Test_Time_min over selected cycles (or all)."""
    dfp = subset_by_cycles(df, cycles=cycles) if cycles is not None else df
    if "Test_Time_min" not in dfp.columns and "Test_Time" in dfp.columns:
        dfp = dfp.copy()
        dfp["Test_Time_min"] = pd.to_numeric(dfp["Test_Time"], errors="coerce") / 60.0

    d = dfp.dropna(subset=["Test_Time_min", "Voltage"])
    d = sort_by_time(d)
    if d.empty:
        print("No Test_Time/Voltage data to plot.")
        return

    V = maybe_flip_voltage(pd.to_numeric(d["Voltage"], errors="coerce").to_numpy(), flipped=flipped)
    plt.figure(figsize=(10, 6))
    plt.plot(d["Test_Time_min"], V, linewidth=lw, label=label)
    plt.xlabel("Test Time (min)", fontsize=18, fontweight='bold')
    plt.ylabel("Voltage (V)", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    if label: plt.legend()
    if label:
        plt.title(f"Voltage vs Time – {label}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compute_ir_and_plot_corrected_profiles(df: pd.DataFrame,
                                           mass_g: float,
                                           cycles=[1,2,50],                        # e.g., [1, 2, 50]
                                           specific_capacity_window_mAhg=(40, 80),
                                           n_points: int = 5,
                                           cap_tol_Ah: float = 1e-6,           # match tolerance in Ah (adjustable)
                                           cccv_cycles=None,                   # cycles using CC–CV phase stitching
                                           step_mode: str = "auto",            # CC-only step selection for other cycles
                                           flipped: bool = False, keep_cap: bool = False,
                                           dQ_min: float = 0.0,
                                           current_min_A: float = 0.0,         # skip points with |I| below this (e.g., to avoid CV tails)
                                           title_suffix: str = "",
                                           verbose: bool = True):
    """
    For selected cycles, compute per-cycle iR correction from a few same-SOC points,
    apply ±delta (charge: -delta, discharge: +delta), and plot corrected V–Q.

    Returns
    -------
    dict: {cycle_index: {"delta_V": delta_med, "R_app_ohm": R_med,
                         "points": [{"Q_mAhg":..., "V_dis":..., "V_chg":..., "dV":..., "I_avg":..., "R":..., "delta":...}, ...],
                         "n_used": N}}
    """
    # Resolve cycles
    selected = list(cycles_list(df, cycles))
    if not selected:
        print("No cycles selected for iR correction.")
        return {}

    # Capacity window in Ah (raw); user gives mAh/g
    q_lo_Ah = (float(specific_capacity_window_mAhg[0]) * float(mass_g)) / 1000.0
    q_hi_Ah = (float(specific_capacity_window_mAhg[1]) * float(mass_g)) / 1000.0
    if q_hi_Ah <= q_lo_Ah:
        raise ValueError("specific_capacity_window_mAhg must be (low, high) with high > low.")

    # Build targets in Ah
    Q_targets = np.linspace(q_lo_Ah, q_hi_Ah, int(n_points))
    cccv_set = set(int(x) for x in (cccv_cycles or []))

    # Colnames
    qcol_dis = select_capacity_column(is_discharge=True,  flipped=flipped, keep_cap=keep_cap)
    qcol_chg = select_capacity_column(is_discharge=False, flipped=flipped, keep_cap=keep_cap)

    out = {}
    # ---------- figure ----------
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap("rainbow", len(selected))

    for i, cyc in enumerate(selected):
        use_phase = (int(cyc) in cccv_set)
        if use_phase:
            dis_df, chg_df = phase_frames_by_cycle(df, cycle=int(cyc), flipped=flipped, dQ_min=dQ_min)
        else:
            dis_df, chg_df = get_step_frames(df, cycle=int(cyc), mode=step_mode)

        # Guard
        if dis_df.empty or chg_df.empty or qcol_dis not in dis_df.columns or qcol_chg not in chg_df.columns:
            print(f"[Cycle {cyc}] Missing step/phase data or capacity columns; skipping.")
            continue

        # Prepare numeric streams
        dis_Q = pd.to_numeric(dis_df[qcol_dis], errors="coerce").to_numpy()
        chg_Q = pd.to_numeric(chg_df[qcol_chg], errors="coerce").to_numpy()
        V_dis = maybe_flip_voltage(pd.to_numeric(dis_df["Voltage"], errors="coerce").to_numpy(), flipped=flipped)
        V_chg = maybe_flip_voltage(pd.to_numeric(chg_df["Voltage"], errors="coerce").to_numpy(), flipped=flipped)

        # Current stream (abs)
        I_dis = np.abs(pd.to_numeric(dis_df.get("Current", pd.Series(index=dis_df.index, dtype=float)), errors="coerce").to_numpy())
        I_chg = np.abs(pd.to_numeric(chg_df.get("Current", pd.Series(index=chg_df.index, dtype=float)), errors="coerce").to_numpy())

        # Helper to find a row within tolerance
        def _nearest_idx(Q_arr, target, tol):
            if Q_arr.size == 0:
                return None
            j = int(np.nanargmin(np.abs(Q_arr - target)))
            return j if np.isfinite(Q_arr[j]) and abs(Q_arr[j] - target) <= tol else None

        # Collect per-point values
        rows = []
        for Qt in Q_targets:
            jd = _nearest_idx(dis_Q, Qt, cap_tol_Ah)
            jc = _nearest_idx(chg_Q, Qt, cap_tol_Ah)
            if jd is None or jc is None:
                if verbose:
                    print(f"[Cycle {cyc}] No matched pair at Q={Qt:.9f} Ah within ±{cap_tol_Ah:g} Ah; skipping.")
                continue

            Id = float(I_dis[jd]) if np.isfinite(I_dis[jd]) else math.nan
            Ic = float(I_chg[jc]) if np.isfinite(I_chg[jc]) else math.nan
            # Skip points with too small current (e.g., CV tails) or NaNs
            if not np.isfinite(Id) or not np.isfinite(Ic):
                continue
            if max(Id, Ic) < float(current_min_A):
                continue

            Iavg = 0.5 * (Id + Ic)
            if Iavg <= 0:
                continue

            Vd = float(V_dis[jd]); Vc = float(V_chg[jc])
            dV = Vc - Vd
            delta = 0.5 * dV
            Rapp = delta / Iavg  # Ω

            rows.append({
                "Q_mAhg": float(capacity_mAhg(Qt, mass_g)),
                "V_dis": Vd, "V_chg": Vc, "dV": dV,
                "I_avg": Iavg, "R_app_ohm": Rapp, "delta_V": delta
            })

        if not rows:
            print(f"[Cycle {cyc}] No valid pairs found in the specified window; skipping correction.")
            continue

        # Robust aggregate (median)
        R_med = float(np.nanmedian([r["R_app_ohm"] for r in rows]))
        delta_med = float(np.nanmedian([r["delta_V"]     for r in rows]))
        out[int(cyc)] = {"delta_V": delta_med, "R_app_ohm": R_med, "points": rows, "n_used": len(rows)}

        if verbose:
            print(f"\n[Cycle {cyc}] iR summary over {len(rows)} points "
                  f"(window={specific_capacity_window_mAhg[0]}–{specific_capacity_window_mAhg[1]} mAh g⁻¹):")
            for r in rows:
                print(f"  Q={r['Q_mAhg']:7.3f} mAh g⁻¹ | dV={1e3*r['dV']:6.1f} mV | Iavg={1e6*r['I_avg']:8.1f} μA "
                      f"| R_app={r['R_app_ohm']:8.2f} Ω | delta={1e3*r['delta_V']:6.1f} mV")
            print(f"  → median R_app = {R_med:.2f} Ω, median delta = {1e3*delta_med:.1f} mV")

        # Plot corrected V–Q for this cycle
        color = cmap(i)

        # Discharge corrected
        Qdis_mAhg = capacity_mAhg(dis_Q, mass_g=mass_g)
        Vdis_corr  = V_dis + delta_med
        plt.plot(Qdis_mAhg, Vdis_corr, lw=1.5, color=color)

        # Charge corrected
        Qchg_mAhg = capacity_mAhg(chg_Q, mass_g=mass_g)
        Vchg_corr  = V_chg - delta_med
        plt.plot(Qchg_mAhg, Vchg_corr, lw=1.5, color=color)

    # Axes & colorbar keyed by cycle
    plt.xlabel("Capacity (mAh g$^{-1}$)", fontsize=18, fontweight='bold')
    plt.ylabel("Voltage (V)",            fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)

    norm = mpl.colors.Normalize(vmin=min(selected), vmax=max(selected))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm); cbar.set_label("Cycle Index", fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    if title_suffix:
        plt.title(f"iR-corrected Voltage Profiles – {title_suffix}", fontsize=16, fontweight='bold')

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    return out



# ---------------------- MAIN ---------------------- #
def main():
    ensure_csv_from_res(res_path, csv_path, parser=(parse_arbinres))
    df = load_frame(csv_path, parse_dates=False)

    # # ---- CE ----
    if ce_use_cycle_wide:
        ce_df = compute_ce_cycle_wide(df, mass_g=mass_g, cycles=cycles_for_ce,
                                      flipped=flipped_polarity, ratio=ce_ratio, method=ce_aggregate_method)
    else:
        ce_df = compute_ce_mixed(df, mass_g=mass_g, cycles=cycles_for_ce, cccv_cycles=cccv_cycles,
                                 step_mode=step_mode, flipped=flipped_polarity, ratio=ce_ratio,
                                 cycle_method=ce_aggregate_method)
    print(ce_df.head(10))
    plot_ce(ce_df)

    # ---- Voltage–capacity (per-cycle mixed) ----
    plot_voltage_profiles_mixed(df, mass_g=mass_g, cycles=cycles_for_profiles,
                                cccv_cycles=cccv_cycles, step_mode=step_mode,
                                flipped=flipped_polarity, keep_cap=keep_cap_unflipped, dQ_min=phase_dQ_min,
                                plotcolor=plotcolor,
                                title_suffix=cellname)

    # # ---- Voltage–time ----
    plot_voltage_vs_time(df, label=cellname, cycles=cycles_for_vt,
                         flipped=flipped_polarity, xlim=vt_xlim, ylim=vt_ylim)
    
    """ # ---- iR correction & corrected V–Q ----
    ir_info = compute_ir_and_plot_corrected_profiles(
    df, mass_g=mass_g,
    cycles=cycles_for_profiles,
    specific_capacity_window_mAhg=(40, 80),
    n_points=5,
    cap_tol_Ah=9e-7,          # make looser if you miss matches, e.g., 5e-6
    cccv_cycles=cccv_cycles,
    step_mode=step_mode,
    flipped=flipped_polarity,
    dQ_min=phase_dQ_min,
    current_min_A=0.0,
    title_suffix=cellname,
    verbose=True)
    deltas_by_cycle = {cyc: v["delta_V"] for cyc, v in ir_info.items()}
    print(deltas_by_cycle)  # e.g., {1: 0.015, 2: 0.012, 50: 0.005} """


if __name__ == "__main__":
    main()
