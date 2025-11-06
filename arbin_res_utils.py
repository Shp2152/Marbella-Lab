
"""
battery_utils.py
----------------
Lightweight helpers for Arbin-style battery CSV workflows.

**Scope (by design):**
- ✅ Data I/O + light prep
- ✅ Cycle and step/phase selection (CC-only and CC–CV aware)
- ✅ Polarity & unit helpers

- ❌ *No* CE% computation here (keep in analysis scripts)
- ❌ *No* dQ/dV utilities here (keep in dedicated module if needed)

This keeps analysis/plot scripts clean while providing a single source of truth
for parsing and selection logic across your repo.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Optional, Union
import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O & light preparation
# ---------------------------------------------------------------------------
def ensure_csv_from_res(res_path: Optional[str],
                        csv_path: str,
                        parser=None) -> None:
    """
    If you sometimes start from an Arbin RES file, this helper will call a
    user-supplied `parser(res_path, csv_path)` function to regenerate the CSV
    when the RES is newer or the CSV is missing. If you only work with CSVs,
    you can ignore this function.

    Parameters
    ----------
    res_path : Optional[str]
        Path to the source .res (or any raw) file. If None, no-op.
    csv_path : str
        Destination CSV path to create/refresh.
    parser : callable or None
        A function with signature `parser(res_path, csv_path)`. Only used if
        provided and `res_path` exists.
    """
    if not res_path or parser is None:
        return
    if not os.path.exists(res_path):
        print(f"[battery_utils] RES not found: {res_path}")
        return
    need_parse = (not os.path.exists(csv_path)) or (
        os.path.getmtime(res_path) > os.path.getmtime(csv_path)
    )
    if need_parse:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        print("[battery_utils] Parsing raw -> CSV ...")
        parser(res_path, csv_path)


def load_frame(csv_path: str,
               parse_dates: bool = False,
               required_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load an Arbin-style CSV and add convenient columns if missing.

    - Adds `Test_Time_min` if `Test_Time` exists and `Test_Time_min` is absent.
    - Leaves all other columns as-is.
    - Optionally parses datetime.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    parse_dates : bool, default False
        If True and a 'DateTime' column exists, attempt to parse it as datetime.
    required_cols : Iterable[str] or None
        If provided, raise ValueError if any of these columns are missing.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    if parse_dates and "DateTime" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            try:
                df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            except Exception:
                pass

    # Ensure minutes column if raw seconds exist
    if "Test_Time_min" not in df.columns and "Test_Time" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["Test_Time_min"] = pd.to_numeric(df["Test_Time"], errors="coerce") / 60.0

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
    return df


# ---------------------------------------------------------------------------
# Generic helpers: cycle filtering & time ordering
# ---------------------------------------------------------------------------
def cycles_list(df: pd.DataFrame,
                cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None) -> List[int]:
    """
    Return a sorted list of cycle indices present in the DataFrame.
    Optionally filter to a subset:
      - cycles=None → all cycles
      - cycles=(lo, hi) → inclusive range [lo..hi]
      - cycles=[c1, c2, ...] or any iterable of ints → only those

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Cycle_Index'.
    cycles : None, tuple, or iterable of ints

    Returns
    -------
    list[int]
    """
    if "Cycle_Index" not in df.columns:
        return []
    all_cycles = (
        pd.to_numeric(df["Cycle_Index"], errors="coerce").dropna()
          .astype(int).unique().tolist()
    )
    all_cycles = sorted(all_cycles)
    if cycles is None:
        return all_cycles
    if isinstance(cycles, tuple) and len(cycles) == 2:
        lo, hi = cycles
        return [c for c in all_cycles if int(lo) <= c <= int(hi)]
    allow = set(int(x) for x in np.atleast_1d(list(cycles)))
    return [c for c in all_cycles if c in allow]


def subset_by_cycles(df: pd.DataFrame,
                     cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None) -> pd.DataFrame:
    """
    Return a view filtered to the selected cycles (or all if cycles is None).
    """
    sel = cycles_list(df, cycles)
    if not sel:
        return df.iloc[0:0] if cycles else df
    return df[df["Cycle_Index"].astype(int).isin(sel)]


def sort_by_time(df: pd.DataFrame,
                 prefer: Iterable[str] = ("Test_Time_min", "Test_Time", "DateTime")) -> pd.DataFrame:
    """Return a copy sorted by the first available time-like column."""
    for c in prefer:
        if c in df.columns:
            key = pd.to_numeric(df[c], errors="coerce") if c != "DateTime" else pd.to_datetime(df[c], errors="coerce")
            return df.loc[key.sort_values(kind="mergesort").index]
    return df.copy()


# ---------------------------------------------------------------------------
# CC-only: step mapping (pick discharge/charge step per cycle)
# ---------------------------------------------------------------------------
def _step_map_minmax(df: pd.DataFrame, cycles: List[int]) -> Dict[int, Tuple[int, int]]:
    """For each cycle: discharge step = min Step_Index; charge step = max Step_Index."""
    out: Dict[int, Tuple[int, int]] = {}
    if "Step_Index" not in df.columns or "Cycle_Index" not in df.columns:
        return out

    w = df.copy()
    w["Step_Index"] = pd.to_numeric(w["Step_Index"], errors="coerce")
    w["Cycle_Index"] = pd.to_numeric(w["Cycle_Index"], errors="coerce")
    w = w[w["Step_Index"].notna() & w["Cycle_Index"].notna()]
    w["Step_Index"] = w["Step_Index"].astype(int)
    w["Cycle_Index"] = w["Cycle_Index"].astype(int)

    for cyc in cycles:
        sub = w[w["Cycle_Index"] == cyc]
        if sub.empty:
            continue
        out[cyc] = (int(sub["Step_Index"].min()), int(sub["Step_Index"].max()))
    return out


def _step_map_auto(df: pd.DataFrame, cycles: List[int]) -> Dict[int, Tuple[int, int]]:
    """
    For each cycle: choose steps with largest increase in the relevant capacity stream.
    (Robust when the CC steps are not simply min/max indices.)
    capacity stream:
        - discharge step → Discharge_Capacity
        - charge step    → Charge_Capacity
    Falls back to min/max if capacity columns are missing.
    """
    out: Dict[int, Tuple[int, int]] = {}
    has_dis = "Discharge_Capacity" in df.columns
    has_chg = "Charge_Capacity" in df.columns
    if "Step_Index" not in df.columns or "Cycle_Index" not in df.columns:
        return out

    w = df.copy()
    w["Step_Index"] = pd.to_numeric(w["Step_Index"], errors="coerce")
    w["Cycle_Index"] = pd.to_numeric(w["Cycle_Index"], errors="coerce")
    w = w[w["Step_Index"].notna() & w["Cycle_Index"].notna()]
    w["Step_Index"] = w["Step_Index"].astype(int)
    w["Cycle_Index"] = w["Cycle_Index"].astype(int)

    for cyc in cycles:
        sub = w[w["Cycle_Index"] == cyc]
        if sub.empty:
            continue
        dis_step = None; chg_step = None
        if has_dis:
            inc = sub.groupby("Step_Index")["Discharge_Capacity"].agg(lambda s: pd.to_numeric(s, errors="coerce").max()
                                                                         - pd.to_numeric(s, errors="coerce").min())
            if len(inc) > 0 and np.isfinite(inc.values).any():
                dis_step = int(inc.fillna(-np.inf).idxmax())
        if has_chg:
            inc = sub.groupby("Step_Index")["Charge_Capacity"].agg(lambda s: pd.to_numeric(s, errors="coerce").max()
                                                                       - pd.to_numeric(s, errors="coerce").min())
            if len(inc) > 0 and np.isfinite(inc.values).any():
                chg_step = int(inc.fillna(-np.inf).idxmax())
        if dis_step is None or chg_step is None:
            smin = int(sub["Step_Index"].min()); smax = int(sub["Step_Index"].max())
            dis_step = smin if dis_step is None else dis_step
            chg_step = smax if chg_step is None else chg_step
        out[cyc] = (int(dis_step), int(chg_step))
    return out


def build_step_map(df: pd.DataFrame,
                   cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None,
                   mode: str = "minmax") -> Dict[int, Tuple[int, int]]:
    """Construct {cycle_index: (discharge_step, charge_step)} using 'minmax' or 'auto'.
    Parameters
    ----------
    df : pd.DataFrame
        Requires 'Cycle_Index' and 'Step_Index'. For mode='auto', also expects
        'Discharge_Capacity' and 'Charge_Capacity' for best results.
    cycles : None | (lo, hi) | iterable[int]
        Which cycles to consider (default all cycles present).
    mode : {'minmax', 'auto'}
        - 'minmax': discharge=min(step), charge=max(step) per cycle.
        - 'auto'  : choose steps with largest capacity increase (robust if
          min/max does not align with the actual CC steps).

    Returns
    -------
    dict[int, tuple[int, int]]
    """
    c = cycles_list(df, cycles)
    if not c:
        return {}
    if mode not in {"minmax", "auto"}:
        raise ValueError("mode must be 'minmax' or 'auto'")
    return _step_map_auto(df, c) if mode == "auto" else _step_map_minmax(df, c)


def get_step_frames(df: pd.DataFrame, cycle: int, mode: str = "minmax") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (discharge_df, charge_df) for a single cycle based on step mapping.
    
    Uses `build_step_map` internally; if the cycle is not found, returns
    (empty_df, empty_df).
    """
    mapping = build_step_map(df, cycles=[cycle], mode=mode)
    if cycle not in mapping:
        empty = df.iloc[0:0]
        return empty, empty
    dis_step, chg_step = mapping[cycle]
    w = df.copy()
    w["Cycle_Index"] = pd.to_numeric(w["Cycle_Index"], errors="coerce").astype("Int64")
    w["Step_Index"] = pd.to_numeric(w["Step_Index"], errors="coerce").astype("Int64")
    dis_df = w[(w["Cycle_Index"] == cycle) & (w["Step_Index"] == dis_step)]
    chg_df = w[(w["Cycle_Index"] == cycle) & (w["Step_Index"] == chg_step)]
    return dis_df, chg_df


# ---------------------------------------------------------------------------
# CC–CV aware: phase selection (stitches CC + CV for each physical phase)
# ---------------------------------------------------------------------------
def phase_frames_by_cycle(df: pd.DataFrame,
                          cycle: int,
                          flipped: bool = False, 
                          keep_cap: bool = False,
                          dQ_min: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (discharge_df, charge_df) for the *entire* cycle by detecting where
    the relevant capacity stream increases. This naturally stitches CC and CV.
    
    
    Parameters
    ----------
    df : DataFrame (must include 'Cycle_Index' and capacity columns)
    cycle : int
        Which cycle to extract.
    flipped : bool
        If True, mapping of physical discharge/charge to capacity columns is flipped.
    dQ_min : float
        Minimum increment (in Ah) to consider "increasing" (filtering noise).

    Returns
    -------
    (dis_df, chg_df) : two DataFrames filtered to rows where the chosen
        capacity stream increases within the cycle, sorted by time.
    """
    if "Cycle_Index" not in df.columns:
        return df.iloc[0:0], df.iloc[0:0]
    cyc = int(cycle)
    w = subset_by_cycles(df, [cyc])
    if w.empty:
        return w.iloc[0:0], w.iloc[0:0]
    w = sort_by_time(w)

    dis_col = select_capacity_column(is_discharge=True,  flipped=flipped, keep_cap=keep_cap)
    chg_col = select_capacity_column(is_discharge=False, flipped=flipped, keep_cap=keep_cap)

    qd = pd.to_numeric(w.get(dis_col, pd.Series(index=w.index, dtype=float)), errors="coerce")
    qc = pd.to_numeric(w.get(chg_col, pd.Series(index=w.index, dtype=float)), errors="coerce")

    dq_dis = qd.diff(); dq_chg = qc.diff()
    is_dis = dq_dis > float(dQ_min)
    is_chg = dq_chg > float(dQ_min)

    dis_df = w[is_dis.fillna(False)].copy()
    chg_df = w[is_chg.fillna(False)].copy()
    return dis_df, chg_df


def iter_phases(df: pd.DataFrame,
                cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None,
                flipped: bool = False,
                keep_cap: bool = False,
                dQ_min: float = 0.0):
    """
    Iterate over selected cycles yielding (cycle, dis_df, chg_df), where each
    DataFrame includes CC and CV parts for that physical phase."""
    for cyc in cycles_list(df, cycles):
        dis, chg = phase_frames_by_cycle(df, cyc, flipped=flipped,keep_cap=keep_cap, dQ_min=dQ_min)
        yield int(cyc), dis, chg


# ---------------------------------------------------------------------------
# Mixed-mode selectors (per-cycle toggle between CC-only vs CC–CV)
# ---------------------------------------------------------------------------
def get_cycle_frames_mixed(df: pd.DataFrame,
                           cycle: int,
                           cccv_enabled: bool,
                           step_mode: str = "auto",
                           flipped: bool = False,
                           keep_cap: bool = False,
                           dQ_min: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (dis_df, chg_df) for a single cycle using CC-only (steps) or CC–CV (phases).

    - If cccv_enabled=True for this cycle → use phase_frames_by_cycle (stitches CC+CV).
    - Else → use get_step_frames with provided step_mode.
    """
    if cccv_enabled:
        return phase_frames_by_cycle(df, cycle, flipped=flipped, keep_cap=keep_cap, dQ_min=dQ_min)
    return get_step_frames(df, cycle, mode=step_mode)


def iter_frames_mixed(df: pd.DataFrame,
                      cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None,
                      cccv_cycles: Optional[Iterable[int]] = None,
                      step_mode: str = "auto",
                      flipped: bool = False, keep_cap: bool = False,
                      dQ_min: float = 0.0):
    """
    Iterate selected cycles, yielding (cycle, dis_df, chg_df) where each cycle
    can independently use CC-only (steps) or CC–CV (phases).

    Parameters
    ----------
    cccv_cycles : iterable[int] or None
        Cycles that should use CC–CV (phase) selection. Others use step selection.
    """
    cccv_set = set(int(x) for x in (cccv_cycles or []))
    for cyc in cycles_list(df, cycles):
        dis, chg = get_cycle_frames_mixed(df, cyc, cccv_enabled=(cyc in cccv_set),
                                          step_mode=step_mode, flipped=flipped, keep_cap=keep_cap, dQ_min=dQ_min)
        yield int(cyc), dis, chg


# ---------------------------------------------------------------------------
# Polarity & unit helpers
# ---------------------------------------------------------------------------
def select_capacity_column(is_discharge: bool, flipped: bool, keep_cap: bool) -> str:
    """Choose capacity column based on physical direction and wiring."""
    if is_discharge:
        if flipped and not keep_cap: #as in if it is flipped but we dont want to keep the original capacity columns then switch if not then keep default
            return "Charge_Capacity"
        else:
            return "Discharge_Capacity"
    else:
        if flipped and not keep_cap: #as in if it is flipped but we dont want to keep the original capacity columns then switch if not then keep default
            return "Discharge_Capacity"
        else:
            return "Charge_Capacity"


def maybe_flip_voltage(V: Union[pd.Series, np.ndarray], flipped: bool):
    """Return -V if flipped else V unchanged (Series or ndarray)."""
    arr = np.asarray(V, dtype=float)
    return -arr if flipped else arr


def capacity_mAhg(Q_Ah: Union[pd.Series, np.ndarray, float], mass_g: float):
    """
    Convert capacity from ampere-hours to mAh per gram:
        mAh/g = (Ah * 1000) / mass_g
    """
    if mass_g is None or mass_g <= 0:
        if isinstance(Q_Ah, (pd.Series, np.ndarray)):
            return np.full(len(np.asarray(Q_Ah)), np.nan)
        return float("nan")
    return (np.asarray(Q_Ah, dtype=float) * 1000.0) / float(mass_g)

# ---------------------------------------------------------------------------
# Ultra Manual step map makers
# ---------------------------------------------------------------------------

# --- Pick the right capacity column for a *step* by whichever increases more ---
def capacity_column_for_step(step_df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    """
    Return 'Charge_Capacity' or 'Discharge_Capacity' by whichever increases more
    within this *single step DataFrame*. Ignores wiring polarity entirely.

    If both spans are equal or non-finite, `prefer` (if 'charge' or 'discharge')
    is used as a tie-breaker; default is 'Charge_Capacity'.
    """
    qc = pd.to_numeric(step_df.get("Charge_Capacity", pd.Series(dtype=float)), errors="coerce")
    qd = pd.to_numeric(step_df.get("Discharge_Capacity", pd.Series(dtype=float)), errors="coerce")
    qc_span = float(qc.max() - qc.min()) if qc.size else float("nan")
    qd_span = float(qd.max() - qd.min()) if qd.size else float("nan")

    if np.isfinite(qc_span) and np.isfinite(qd_span):
        if qc_span > qd_span: return "Charge_Capacity"
        if qd_span > qc_span: return "Discharge_Capacity"
    # tie or NaNs
    if prefer and str(prefer).lower().startswith("dis"):
        return "Discharge_Capacity"
    return "Charge_Capacity"


# --- Optional manual step-map  ---
def build_step_map_manual(df: pd.DataFrame,
                          user_spec: Dict[Union[int, str], Dict[str, Union[int, Iterable[int]]]],
                          cycles: Optional[Union[Tuple[int, int], Iterable[int]]] = None
                          ) -> Dict[int, Tuple[int, int]]:
    """
    Build {cycle: (discharge_step, charge_step)} from a user dictionary.

    user_spec examples:
        {"default": {"charge": 4, "discharge": 5}}
        {1: {"charge": 2, "discharge": 1}, "default": {"charge": 4, "discharge": 5}}

    - Values may be int or list[int]; if list, the first element is used.
    - Unknown cycles fall back to 'default' if provided; otherwise omitted.
    """
    sel = cycles_list(df, cycles)
    out: Dict[int, Tuple[int, int]] = {}

    def _first_int(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            for v in x:
                if v is not None: return int(v)
            return None
        return None if x is None else int(x)

    default = user_spec.get("default", {})
    for cyc in sel:
        spec = user_spec.get(int(cyc), default)
        ds = _first_int(spec.get("discharge", None))
        cs = _first_int(spec.get("charge", None))
        if ds is None or cs is None:
            continue
        out[int(cyc)] = (int(ds), int(cs))
    return out

def normalize_manual_steps(df: pd.DataFrame,
                           user_spec: dict,
                           cycles=None) -> dict[int, tuple[list[int], list[int]]]:
    """
    user_spec example:
        {"default": {"discharge": [1,4], "charge": [2,5]},
         7: {"discharge": [2,3], "charge": [4]}}   # cycle 7 override

    Returns {cycle: ([dis_steps...], [chg_steps...])}, filtering to steps that exist.
    """
    def _to_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple, set)): return [int(v) for v in x]
        return [int(x)]

    selected = cycles_list(df, cycles)  # already in this module
    w = df.copy()
    w["Cycle_Index"] = pd.to_numeric(w["Cycle_Index"], errors="coerce").astype("Int64")
    w["Step_Index"]  = pd.to_numeric(w["Step_Index"],  errors="coerce").astype("Int64")

    out = {}
    default = user_spec.get("default", {})
    for cyc in selected:
        spec = user_spec.get(int(cyc), default)
        dis_steps = _to_list(spec.get("discharge", []))
        chg_steps = _to_list(spec.get("charge", []))
        # keep only steps that actually appear in this cycle
        existing = set(w.loc[w["Cycle_Index"]==int(cyc), "Step_Index"].dropna().astype(int).unique())
        dis = [s for s in dis_steps if s in existing]
        chg = [s for s in chg_steps if s in existing]
        out[int(cyc)] = (dis, chg)
    return out


# Aliases (optional)
as_mAhg = capacity_mAhg
flip_voltage = maybe_flip_voltage


# ---------------------------------------------------------------------------
# NEW: Cycle-level terminal capacities & CE extraction
# ---------------------------------------------------------------------------
def cycle_terminal_capacities(df: pd.DataFrame,
                              cycles=None,
                              flipped: bool = False,
                              keep_cap: bool = False,
                              method: str = "max") -> pd.DataFrame:
    """
    Compute per-cycle terminal capacities from an Arbin-style frame.

    Parameters
    ----------
    df : DataFrame
        Must include 'Cycle_Index' and capacity columns.
    cycles : None | (lo, hi) | iterable[int]
        Which cycles to include.
    flipped : bool
        If True, swap charge/discharge capacity columns logically (for reversed wiring).
    keep_cap : bool
        If True with `flipped`, do not swap column selection (rare). See `select_capacity_column`.
    method : {"max","span"}
        - "max"  : use max value of the capacity stream within the cycle
        - "span" : use (max - min) within the cycle (robust vs reset offsets)

    Returns
    -------
    DataFrame with columns: ["Cycle_Index", "Qdis_Ah", "Qchg_Ah"]
    """
    rows = []
    for cyc in cycles_list(df, cycles):
        w = df[pd.to_numeric(df.get("Cycle_Index", pd.Series(dtype=float)), errors="coerce").astype("Int64") == int(cyc)]
        if w.empty:
            continue
        qd_col = select_capacity_column(True, flipped, keep_cap)
        qc_col = select_capacity_column(False, flipped, keep_cap)

        qd = pd.to_numeric(w.get(qd_col, pd.Series(index=w.index, dtype=float)), errors="coerce").dropna()
        qc = pd.to_numeric(w.get(qc_col, pd.Series(index=w.index, dtype=float)), errors="coerce").dropna()
        if qd.empty or qc.empty:
            continue

        if str(method).lower() == "span":
            Qdis = float(qd.max() - qd.min())
            Qchg = float(qc.max() - qc.min())
        else:
            Qdis = float(qd.max())
            Qchg = float(qc.max())

        rows.append({"Cycle_Index": int(cyc), "Qdis_Ah": Qdis, "Qchg_Ah": Qchg})
    return pd.DataFrame(rows).sort_values("Cycle_Index").reset_index(drop=True)


def cycle_specific_capacities(df: pd.DataFrame,
                              mass_g: float,
                              cycles=None,
                              flipped: bool = False,
                              keep_cap: bool = False,
                              method: str = "max") -> pd.DataFrame:
    """
    Same as `cycle_terminal_capacities`, but also returns specific capacities (mAh/g).
    Columns added: "Qdis_mAhg", "Qchg_mAhg".
    """
    base = cycle_terminal_capacities(df, cycles=cycles, flipped=flipped, keep_cap=keep_cap, method=method)
    if base.empty:
        return base
    base = base.copy()
    base["Qdis_mAhg"] = capacity_mAhg(base["Qdis_Ah"].to_numpy(), mass_g=mass_g)
    base["Qchg_mAhg"] = capacity_mAhg(base["Qchg_Ah"].to_numpy(), mass_g=mass_g)
    return base


def cycle_ce_from_termcaps(term_df: pd.DataFrame, ratio: str = "dis_over_chg") -> pd.DataFrame:
    """
    Compute CE% from a DataFrame produced by `cycle_terminal_capacities`.
    Returns columns: ["Cycle_Index","CE_percent"].
    """
    if term_df is None or term_df.empty:
        return term_df.iloc[0:0]
    w = term_df.copy()
    r = str(ratio).lower()
    # CE is dimensionless; mass cancels, so Ah is fine
    with pd.option_context("mode.chained_assignment", None):
        if r.startswith("chg") or r == "chg_over_dis":
            w["CE_percent"] = (pd.to_numeric(w["Qchg_Ah"], errors="coerce") /
                               pd.to_numeric(w["Qdis_Ah"], errors="coerce")) * 100.0
        else:
            w["CE_percent"] = (pd.to_numeric(w["Qdis_Ah"], errors="coerce") /
                               pd.to_numeric(w["Qchg_Ah"], errors="coerce")) * 100.0
    return w[["Cycle_Index", "CE_percent"]].sort_values("Cycle_Index").reset_index(drop=True)


def cycle_ce(df: pd.DataFrame,
             cycles=None,
             flipped: bool = False, keep_cap: bool = False,
             method: str = "max",
             ratio: str = "dis_over_chg") -> pd.DataFrame:
    """
    Convenience: compute CE% directly from a raw frame.
    """
    term = cycle_terminal_capacities(df, cycles=cycles, flipped=flipped, keep_cap=keep_cap, method=method)
    return cycle_ce_from_termcaps(term, ratio=ratio)


def extract_cycle_terminal_capacities_from_csv(csv_path: str,
                                               cycles=None,
                                               flipped: bool = False,
                                               keep_cap: bool = False,
                                               method: str = "max",
                                               mass_g: float = None) -> pd.DataFrame:
    """
    One-shot helper: load CSV and return terminal capacities per cycle.
    Adds mAh/g columns if `mass_g` is provided (>0).
    """
    df = load_frame(csv_path, parse_dates=False)
    base = cycle_terminal_capacities(df, cycles=cycles, flipped=flipped, keep_cap=keep_cap, method=method)
    if mass_g and float(mass_g) > 0:
        base = base.copy()
        base["Qdis_mAhg"] = capacity_mAhg(base["Qdis_Ah"].to_numpy(), mass_g=float(mass_g))
        base["Qchg_mAhg"] = capacity_mAhg(base["Qchg_Ah"].to_numpy(), mass_g=float(mass_g))
    return base
