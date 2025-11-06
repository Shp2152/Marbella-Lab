# -*- coding: utf-8 -*-
"""
ica_foi.py
----------
Feature-of-interest (FOI) extraction for ICA (dQ/dV) curves.

Inputs:
    V : 1D array of voltage (V), increasing
    Y : 1D array of dQ/dV (mAh / (V·g)), same length as V

Outputs:
    FOIs per branch: onset, peak position & height, FWHM, delta_max (=Vmax - Vonset)
    and convenience helpers for cycle-to-cycle and charge-vs-discharge diffs.

Notes:
- "Onset" is computed with a noise-aware absolute threshold (Method A):
  threshold = max(abs_floor, k_sigma * baseline_sigma),
  where baseline_sigma is measured in a user-specified pre-front window.
- FWHM uses 0 baseline and finds half-height crossings by linear interpolation.
- All functions are branch-agnostic; set `sign=+1` for positive peaks (charge),
  `sign=-1` for negative peaks (discharge) when both_positive=False.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks


# ----------------------------- utilities -----------------------------
def _robust_sigma(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    return 1.4826 * mad  # MAD -> sigma


def _interp_x_for_y(x0, y0, x1, y1, y_target):
    # Linear interpolation for y_target between (x0,y0) and (x1,y1)
    if not np.isfinite(x0) or not np.isfinite(x1) or y1 == y0:
        return x1
    return x0 + (y_target - y0) * (x1 - x0) / (y1 - y0)


# ----------------------------- FOIs -----------------------------
def onset_by_noise_threshold(
    V: np.ndarray,
    Y: np.ndarray,
    sign: int,
    pre_window: Optional[Tuple[float, float]],
    k_sigma: float = 5.0,
    abs_floor: float = 800.0,
    side: str = "left",             # NEW: 'left' (charge) or 'right' (discharge)
    idx_peak: Optional[int] = None  # NEW: index of main peak (in sign*Y)
) -> Tuple[float, Dict]:
    """
    Method A (noise-aware absolute threshold) with side-aware bracketing
    near the main peak. Returns (Vonset, meta).
    """
    V = np.asarray(V, float); Y = np.asarray(Y, float)
    if V.size < 4 or Y.size < 4:
        return float("nan"), {"threshold": np.nan, "sigma": np.nan, "idx": None, "success": False}

    # --- 1) Baseline noise from pre-front window; fallback to first ~10% bins
    if pre_window is not None:
        m = (V >= float(pre_window[0])) & (V <= float(pre_window[1]))
        base = Y[m]
    else:
        base = np.array([])
    if base.size < 5:
        n_fallback = max(5, int(0.10 * V.size))
        base = Y[:n_fallback]

    sigma = _robust_sigma(base)
    T0 = max(float(abs_floor), float(k_sigma) * float(sigma))
    y = sign * Y  # positive shape for both branches

    # --- 2) Peak index (in y)
    if idx_peak is None:
        with np.errstate(invalid="ignore"):
            idx_peak = int(np.nanargmax(y))

    # Crossing sets (vectorized):
    #   rising (left edge):   y[k] < T <= y[k+1]
    #   falling (right edge): y[k] >= T > y[k+1]
    def rising_idxs(y, T):
        return np.where((y[:-1] < T) & (y[1:] >= T))[0]
    def falling_idxs(y, T):
        return np.where((y[:-1] >= T) & (y[1:] < T))[0]

    # --- 3) Search at T0 near the main peak, on the requested side
    def _pick_crossing(y, T, side, idx_pk):
        if side == "left":
            cand = rising_idxs(y, T)
            cand = cand[cand < idx_pk]        # before the peak
            return (cand[-1] if cand.size else None, "rising")
        else:
            cand = falling_idxs(y, T)
            cand = cand[cand >= idx_pk]       # after (or at) the peak
            return (cand[0] if cand.size else None, "falling")

    k, mode = _pick_crossing(y, T0, side, idx_peak)

    # relax threshold a bit if needed
    if k is None:
        for scale in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60):
            T_try = scale * T0
            k, mode = _pick_crossing(y, T_try, side, idx_peak)
            if k is not None:
                T0 = T_try
                break

    if k is None:
        return float("nan"), {"threshold": T0, "sigma": sigma, "idx": None, "success": False}

    # --- 4) Interpolate inside the bracket where the crossing occurs
    if mode == "rising":  # between k and k+1
        Von = _interp_x_for_y(V[k], y[k], V[k+1], y[k+1], T0)
        idx = k + 1
    else:                 # "falling": between k and k+1 as well
        Von = _interp_x_for_y(V[k], y[k], V[k+1], y[k+1], T0)
        idx = k + 1

    Von = float(np.clip(Von, V.min(), V.max()))
    return Von, {"threshold": T0, "sigma": sigma, "idx": idx, "success": True}


def peak_max(
    V: np.ndarray,
    Y: np.ndarray,
    sign: int,
    search_window: Optional[Tuple[float, float]] = None,
    prominence: Optional[float] = None,
    width: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    Find peak with scipy.signal.find_peaks on sign*Y.

    Returns
    -------
    Vmax : float (np.nan if not found)
    H    : float (peak height in the original sign convention: positive for charge,
                  negative for discharge when sign=-1 and both_positive=False)
    idx  : int   (index of the peak, or -1)
    """
    V = np.asarray(V, float)
    ysig = sign * np.asarray(Y, float)

    mask = np.ones_like(V, dtype=bool)
    if search_window is not None:
        mask = (V >= search_window[0]) & (V <= search_window[1])

    if not np.any(mask):
        return float("nan"), float("nan"), -1

    yview = ysig[mask]
    peaks, props = find_peaks(yview, prominence=prominence, width=width)

    if peaks.size == 0:
        return float("nan"), float("nan"), -1

    # choose most prominent (or tallest)
    choose = np.argmax(props["prominences"] if "prominences" in props else yview[peaks])
    idx_rel = int(peaks[choose])
    full_idx = int(np.where(mask)[0][idx_rel])

    H_signed = (sign * ysig[full_idx])  # = original Y[full_idx]
    return float(V[full_idx]), float(H_signed), full_idx


def fwhm(
    V: np.ndarray,
    Y: np.ndarray,
    sign: int,
    idx_peak: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Full width at half maximum by linear interpolation around the peak.
    Assumes baseline ~ 0.
    Returns (FWHM, V_left_half, V_right_half). np.nan if not computable.
    """
    V = np.asarray(V, float); Y = np.asarray(Y, float)
    ysig = sign * Y  # positive peak shape
    if idx_peak is None:
        # fallback: take global maximum of ysig
        idx_peak = int(np.nanargmax(ysig))
    if not np.isfinite(ysig[idx_peak]) or ysig[idx_peak] <= 0:
        return float("nan"), float("nan"), float("nan")

    half = 0.5 * ysig[idx_peak]

    # search left crossing
    iL = None
    for i in range(idx_peak, 0, -1):
        if ysig[i - 1] <= half <= ysig[i]:
            iL = i
            break
    # search right crossing
    iR = None
    for i in range(idx_peak, V.size - 1):
        if ysig[i] >= half >= ysig[i + 1]:
            iR = i + 1
            break

    if iL is None or iR is None:
        return float("nan"), float("nan"), float("nan")

    VL = _interp_x_for_y(V[iL - 1], ysig[iL - 1], V[iL], ysig[iL], half)
    VR = _interp_x_for_y(V[iR - 1], ysig[iR - 1], V[iR], ysig[iR], half)
    return float(VR - VL), float(VL), float(VR)


def branch_foi(
    V: np.ndarray,
    Y: np.ndarray,
    sign: int,
    *,
    onset_pre_window: Tuple[float, float],
    onset_k_sigma: float = 5.0,
    onset_abs_floor: float = 800.0,
    onset_min_consecutive: int = 3,      # kept for API compatibility
    peak_search_window: Optional[Tuple[float, float]] = None,
    peak_prominence: Optional[float] = None,
    front_side: str = "left",            # NEW: 'left' or 'right'
) -> Dict[str, float]:
    # 1) Peak first (on sign*Y)
    Vmax, H_signed, idx = peak_max(
        V, Y, sign=sign, search_window=peak_search_window, prominence=peak_prominence
    )

    # 2) Onset at the requested side of the main peak
    Von, meta = onset_by_noise_threshold(
        V, Y, sign=sign, pre_window=onset_pre_window,
        k_sigma=onset_k_sigma, abs_floor=onset_abs_floor,
        side=front_side, idx_peak=idx
    )

    if idx < 0 or not np.isfinite(Vmax):
        return {
            "Vonset": float(Von), "Vmax": float("nan"),
            "Hmax_abs": float("nan"), "Hmax_signed": float("nan"),
            "FWHM": float("nan"), "delta_max": float("nan")
        }

    # 3) FWHM on the positive shape
    W, _, _ = fwhm(V, Y, sign=sign, idx_peak=idx)
    dmax = (Vmax - Von) if np.isfinite(Von) else float("nan")

    return {
        "Vonset": float(Von),
        "Vmax": float(Vmax),
        "Hmax_abs": float(abs(H_signed)),
        "Hmax_signed": float(H_signed),
        "FWHM": float(W),
        "delta_max": float(dmax),
    }


def cycle_foi_row(
    V_dis: np.ndarray, Y_dis: np.ndarray,
    V_chg: np.ndarray, Y_chg: np.ndarray,
    *,
    prewin_dis: Tuple[float, float],
    prewin_chg: Tuple[float, float],
    k_sigma: float = 5.0,
    abs_floor: float = 800.0,
    min_consecutive: int = 3,
    both_positive: bool = False,
    peak_window: Optional[Tuple[float, float]] = None,
    peak_prominence: Optional[float] = None,
    cycle_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute FOIs for charge and discharge for a cycle and add combined diffs.
    """
    sign_dis = +1 if both_positive else -1   # discharge is negative if not flipped
    sign_chg = +1

    dis = branch_foi(
        V_dis, Y_dis, sign_dis,
        onset_pre_window=prewin_dis, onset_k_sigma=k_sigma, onset_abs_floor=abs_floor,
        onset_min_consecutive=min_consecutive, peak_search_window=peak_window,
        peak_prominence=peak_prominence, front_side="right"   # <- discharge onset = right-hand edge
    )
    chg = branch_foi(
        V_chg, Y_chg, sign_chg,
        onset_pre_window=prewin_chg, onset_k_sigma=k_sigma, onset_abs_floor=abs_floor,
        onset_min_consecutive=min_consecutive, peak_search_window=peak_window,
        peak_prominence=peak_prominence, front_side="left"    # <- charge onset = left-hand edge
    )

    row = {
        "cycle": int(cycle_index) if cycle_index is not None else -1,
        # Onsets
        "Vonset_dis": dis["Vonset"], "Vonset_chg": chg["Vonset"],
        # Peak positions & heights
        "Vmax_dis": dis["Vmax"], "Vmax_chg": chg["Vmax"],
        "Hmax_dis_abs": dis["Hmax_abs"], "Hmax_chg_abs": chg["Hmax_abs"],
        "Hmax_dis_signed": dis["Hmax_signed"], "Hmax_chg_signed": chg["Hmax_signed"],
        # Widths
        "FWHM_dis": dis["FWHM"], "FWHM_chg": chg["FWHM"],
        # delta_max
        "delta_max_dis": dis["delta_max"], "delta_max_chg": chg["delta_max"],
        # charge–discharge Vmax difference this cycle
        "Vmax_chg_minus_dis": (chg["Vmax"] - dis["Vmax"]) if np.isfinite(chg["Vmax"]) and np.isfinite(dis["Vmax"]) else np.nan,
    }
    return row


def vpeak_shift_between_cycles(df_rows: Dict[int, Dict[str, float]], c1: int, c2: int) -> Dict[str, float]:
    """
    Convenience helper: V_peak,max shifts (charge and discharge) between two cycles.
    """
    r1 = df_rows.get(int(c1), {})
    r2 = df_rows.get(int(c2), {})
    out = {}
    if "Vmax_chg" in r1 and "Vmax_chg" in r2:
        out["Vmax_chg_c1_minus_c2"] = r1["Vmax_chg"] - r2["Vmax_chg"]
    else:
        out["Vmax_chg_c1_minus_c2"] = np.nan
    if "Vmax_dis" in r1 and "Vmax_dis" in r2:
        out["Vmax_dis_c1_minus_c2"] = r1["Vmax_dis"] - r2["Vmax_dis"]
    else:
        out["Vmax_dis_c1_minus_c2"] = np.nan
    return out
