"""Compare local-branch refit schedules without changing the frozen V4 model."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "minjiang-v4"
sys.path.insert(0, str(V4))
import candidate  # noqa: E402


DAY = pd.Timedelta(days=1)
PERIODS = (("2021", "2021-01-01", "2021-12-31"),
           ("historical_2022_23", "2022-01-01", "2023-12-31"))
OPTIONS = (("annual", None), ("annual", 6), ("annual", 4),
           ("quarterly", None), ("quarterly", 6), ("quarterly", 4))
HIGH_FLOW = 3602.715


def metrics(target, prediction):
    error = prediction - target
    high = target >= HIGH_FLOW
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(error.abs().mean()),
                high_rmse=float(np.sqrt(np.mean(error[high]**2))))


def local_forecast(tables, first, last, cadence, years, config):
    full, _, _, _, q = tables
    target = q.shift(-1)
    target_dates = q.index + DAY
    starts = {pd.Timestamp(first)}
    frequency = "YS" if cadence == "annual" else "QS"
    starts.update(pd.date_range(first, last, freq=frequency))
    starts = sorted(starts)
    pieces = []
    for i, start in enumerate(starts):
        stop = min(pd.Timestamp(last), starts[i + 1] - DAY) if i + 1 < len(starts) else pd.Timestamp(last)
        train = (full.notna().all(axis=1) & target.notna() &
                 (target_dates >= pd.Timestamp("2014-02-01")) & (target_dates < start))
        if years is not None:
            train &= target_dates >= start - pd.DateOffset(years=years)
        evaluate = (target_dates >= start) & (target_dates <= stop)
        if train.sum() < 365 or not full.notna().all(axis=1)[evaluate].all():
            raise ValueError(f"Incomplete fit or forecast inputs for {start.date()}")
        _, predicted = next(candidate.predict_options(tables, train, evaluate, target, [config]))
        predicted.index += DAY
        pieces.append(predicted)
    forecast = pd.concat(pieces).rename("local")
    if not forecast.index.equals(pd.date_range(first, last)) or not np.isfinite(forecast).all():
        raise ValueError("Incomplete or misaligned local forecasts")
    return forecast


def evaluate_period(data, name, first, last, config, options):
    reference = pd.read_csv(V4 / "results" / ("outer_2021.csv" if name == "2021" else
                                               "test_2022_23.csv"), index_col="target_date", parse_dates=True)
    reference = reference.loc[first:last]
    if not reference.index.equals(pd.date_range(first, last)):
        raise ValueError("Reference dates are incomplete")
    tables = candidate.inputs(data, pd.Timestamp(last) - DAY)
    result = [dict(period=name, model="frozen_v4", **metrics(reference.target, reference.prediction))]
    for cadence, years in options:
        label = f"{cadence}_{years or 'all'}"
        local = local_forecast(tables, first, last, cadence, years, config)
        if cadence == "annual" and years is None:
            initial = local.loc[first: min(last, str(pd.Timestamp(first).year) + "-12-31")]
            np.testing.assert_allclose(initial, reference.local.loc[initial.index], atol=1e-6, rtol=0)
        forecast = .5 * (reference.prior + local)
        result.append(dict(period=name, model=label, **metrics(reference.target, forecast)))
    return pd.DataFrame(result)


def main():
    data = ROOT / "minjiang-original" / "data"
    if not data.exists():
        raise FileNotFoundError("Run minjiang-v6/reproduce.py first to prepare the verified data")
    choice = json.loads((V4 / "candidate_results" / "candidate_choice.json").read_text(encoding="utf-8"))
    config = (choice["alpha"], choice["neighbors"], choice["penalty"])
    development = evaluate_period(data, *PERIODS[0], config, OPTIONS)
    baseline = development.iloc[0]
    candidates = development.iloc[1:].copy()
    eligible = candidates.loc[(candidates.rmse <= .99 * baseline.rmse) &
                              (candidates.mae <= 1.03 * baseline.mae) &
                              (candidates.high_rmse <= 1.05 * baseline.high_rmse)]
    selected = "frozen_v4" if eligible.empty else eligible.sort_values("rmse").iloc[0].model
    print("2021 validation:")
    print(development.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Selected from 2021:", selected, flush=True)
    if selected != "frozen_v4":
        options = [option for option in OPTIONS if f"{option[0]}_{option[1] or 'all'}" == selected]
        historical = evaluate_period(data, *PERIODS[1], config, options)
        print("2022-2023 historical comparison (not a new blind test):")
        print(historical.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
