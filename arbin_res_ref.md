Function mini‑reference for "arbin_res_utils.py"

load_frame(path, parse_dates=False, required_cols=None)
Reads CSV; adds Test_Time_min if you have Test_Time (interpreted as seconds).

cycles_list(df, cycles=None)
Returns sorted list of cycle indices. Supports None, (lo, hi), or explicit iterables.

subset_by_cycles(df, cycles=None)
Convenience filter for cycles.

build_step_map(df, cycles=None, mode='minmax'|'auto')
Returns {cycle: (dis_step, chg_step)}.

'minmax': discharge = min Step_Index, charge = max.

'auto' : picks the step with largest increase in the relevant capacity stream.

get_step_frames(df, cycle, mode='minmax')
Returns two DataFrames: (discharge_df, charge_df) for that cycle.

iter_selected_steps(df, cycles=None, mode='minmax')
Yields (cycle, discharge_df, charge_df) across selected cycles.

select_capacity_column(is_discharge, flipped)
Chooses the correct capacity column name given your wiring.

maybe_flip_voltage(V, flipped)
Flips voltage sign if you wired with reversed polarity.

capacity_mAhg(Q_Ah, mass_g) (alias: as_mAhg)
Converts Ah → mAh/g (vectorized).


**Example code:**
from battery_utils import (
    load_frame, cycles_list, subset_by_cycles,
    build_step_map, get_step_frames, iter_selected_steps,
    select_capacity_column, maybe_flip_voltage, capacity_mAhg
)

df = load_frame("LTOc288.csv", parse_dates=False)

# Choose cycles and build a (dis_step, chg_step) map per cycle
step_map = build_step_map(df, cycles=(1, 50), mode="auto")

# Iterate cycles and grab the specific discharge/charge step frames
for cyc, dis_df, chg_df in iter_selected_steps(df, cycles=(1, 10), mode="auto"):
    # Example: make V–Q (mAh/g) traces for each selected step
    # Discharge:
    qcol_dis = select_capacity_column(is_discharge=True,  flipped=False)
    Qdis_mAhg = capacity_mAhg(dis_df[qcol_dis].to_numpy(), mass_g=0.004)
    Vdis = maybe_flip_voltage(dis_df["Voltage"], flipped=False)

    # Charge:
    qcol_chg = select_capacity_column(is_discharge=False, flipped=False)
    Qchg_mAhg = capacity_mAhg(chg_df[qcol_chg].to_numpy(), mass_g=0.004)
    Vchg = maybe_flip_voltage(chg_df["Voltage"], flipped=False)

    # ...plot your curves here...
