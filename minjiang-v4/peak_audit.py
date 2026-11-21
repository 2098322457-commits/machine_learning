"""Compare daily Nanping forecasts on observed high-flow days using one training threshold."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DAY = pd.Timedelta(days=1)
TRAIN_START, TRAIN_END = "2014-02-01", "2020-12-31"
MODELS = {"group": "group", "v4": "prediction"}


def rmse(errors):
    return float(np.sqrt(np.square(errors).mean()))


def high_flow_events(frame, threshold):
    """Strict consecutive runs. Split intervening low days at the midpoint for timing windows."""
    active = frame.target.ge(threshold)
    starts = frame.index[active & ~active.shift(fill_value=False)]
    ends = frame.index[active & ~active.shift(-1, fill_value=False)]
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        # Timing may move up to two days outside the observed run. When two windows
        # meet, the nearest run gets the date; ties go to the earlier run.
        window_start = max(frame.index[0], start - 2*DAY)
        window_end = min(frame.index[-1], end + 2*DAY)
        if i:
            gap = (start - ends[i-1]).days
            window_start = max(window_start, ends[i-1] + (gap//2 + 1)*DAY)
        if i + 1 < len(starts):
            gap = (starts[i+1] - end).days
            window_end = min(window_end, end + (gap//2)*DAY)
        yield start, end, window_start, window_end


def evaluate_period(path, period, start, stop, truth, threshold):
    frame = pd.read_csv(path, parse_dates=["target_date"]).set_index("target_date", verify_integrity=True)
    if not frame.index.equals(pd.date_range(start, stop, freq="D")):
        raise ValueError(f"Unexpected evaluation dates in {path}")
    np.testing.assert_allclose(frame.target, truth.loc[frame.index], rtol=0, atol=1e-9)
    if not np.isfinite(frame[["target", *MODELS.values()]]).all().all():
        raise ValueError(f"Non-finite forecast or target in {path}")
    above = frame.target.ge(threshold)
    rows = []
    multi_days = pd.Series(False, index=frame.index)
    for start, end, left, right in high_flow_events(frame, threshold):
        core, window = frame.loc[start:end], frame.loc[left:right]
        peak_date = core.target.idxmax()
        row = dict(period=period, threshold=threshold, start=start, end=end, n_high_days=len(core),
                   window_start=left, window_end=right, true_peak_date=peak_date, true_peak=core.target.max())
        if len(core) >= 2:
            multi_days.loc[start:end] = True
        for name, col in MODELS.items():
            predicted_peak_date = window[col].idxmax()
            predicted_peak = window[col].loc[predicted_peak_date]
            row.update({f"{name}_rmse": rmse(core[col] - core.target),
                        f"{name}_peak_date": predicted_peak_date, f"{name}_peak": predicted_peak,
                        f"{name}_peak_bias": predicted_peak - row["true_peak"],
                        f"{name}_offset_days": (predicted_peak_date-peak_date).days,
                        f"{name}_error_on_true_peak_day": core.at[peak_date, col]-core.at[peak_date, "target"]})
        rows.append(row)

    print(f"{period}: {above.sum()} high days / {len(frame)} days, {len(rows)} events; "
          f"{sum(r['n_high_days'] >= 2 for r in rows)} multi-day events / {multi_days.sum()} days.")
    if rows:
        table = pd.DataFrame(rows)
        if table.n_high_days.sum() != int(above.sum()):
            raise ValueError("High-flow days counted twice or omitted")
        if (table.window_end.shift() >= table.window_start).any():
            raise ValueError("Event timing windows overlap")
        biggest = table.loc[table.true_peak.idxmax()]
        print(f"Maximum observed {biggest.true_peak:.2f} on {biggest.true_peak_date.date()}.")
        for name, col in MODELS.items():
            error = frame[col] - frame.target
            offsets = table[name + "_offset_days"]
            print(f"  {name} RMSE over high days {rmse(error[above]):.6f}; "
                  f"over multi-day high days {rmse(error[multi_days]):.6f}; "
                  f"peak absolute bias mean {table[name+'_peak_bias'].abs().mean():.6f}; "
                  f"mean absolute offset {offsets.abs().mean():.3f} days; "
                  f"correct peak dates {offsets.eq(0).sum()}/{len(rows)}; "
                  f"largest event bias {biggest[name+'_peak_bias']:+.3f}, offset {biggest[name+'_offset_days']:+d}.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent/"minjiang-original"/"data")
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    raw = pd.read_csv(args.data/"glofas_daily_discharge.csv", parse_dates=["date"])
    truth = raw.loc[raw.station.eq("nanping")].set_index("date", verify_integrity=True).river_discharge_m3s
    training = truth.loc[TRAIN_START:TRAIN_END]
    if not training.index.equals(pd.date_range(TRAIN_START, TRAIN_END)) or not np.isfinite(training).all():
        raise ValueError("Incomplete training reference period")
    threshold = float(training.quantile(0.95, interpolation="linear"))
    print(f"{len(training)} eligible training days, fixed 95th percentile {threshold:.6f} m3/s.")
    rows = []
    for period, start, stop in [("outer_2021", "2021-01-01", "2021-12-31"),
                                ("test_2022_23", "2022-01-01", "2023-12-31")]:
        rows.extend(evaluate_period(args.out/(period+".csv"), period, start, stop, truth, threshold))
    if not rows:
        raise ValueError("No observed high-flow events found")
    pd.DataFrame(rows).to_csv(args.out/"peak_events.csv", index=False, date_format="%Y-%m-%d")


if __name__ == "__main__":
    main()
