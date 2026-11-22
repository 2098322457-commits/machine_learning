"""Select a fixed branch blend on 2021, then compare it with V4 historically."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "minjiang-v4" / "results"
HIGH_FLOW = 3602.715
WEIGHTS = {
    "frozen_v4": (.5, .5, 0.),
    "balanced_ar_25": (.375, .375, .25),
    "local_50_ar_25": (.25, .5, .25),
    "prior_50_ar_25": (.5, .25, .25),
    "prior_25_ar_50": (.25, .25, .5),
    "local_75_ar_25": (0., .75, .25),
    "local_50_ar_50": (0., .5, .5),
}


def read(name, start, end):
    frame = pd.read_csv(RESULTS / name, index_col="target_date", parse_dates=True)
    if not frame.index.equals(pd.date_range(start, end, name="target_date")):
        raise ValueError(f"Incomplete reference dates in {name}")
    if not np.isfinite(frame[["target", "prior", "local", "ar", "prediction"]]).all().all():
        raise ValueError(f"Missing target or branch prediction in {name}")
    np.testing.assert_allclose(frame.prediction, .5 * (frame.prior + frame.local), atol=1e-8, rtol=0)
    return frame


def score(frame, weights):
    prediction = frame[["prior", "local", "ar"]].to_numpy() @ np.asarray(weights)
    error = prediction - frame.target.to_numpy()
    peak = frame.target.to_numpy() >= HIGH_FLOW
    return dict(n=len(frame), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))),
                high_rmse=float(np.sqrt(np.mean(error[peak]**2))))


def gain_interval(frame, weights):
    old = frame.prediction.to_numpy() - frame.target.to_numpy()
    new = frame[["prior", "local", "ar"]].to_numpy() @ np.asarray(weights) - frame.target.to_numpy()
    rng = np.random.default_rng(2026)
    n, block, repetitions = len(frame), 14, 4000
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n / block)), 1))
    indices = ((starts + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    improvement = np.sqrt(np.mean(old[indices]**2, axis=1)) - np.sqrt(np.mean(new[indices]**2, axis=1))
    return np.quantile(improvement, [.025, .975])


def main():
    validation = read("outer_2021.csv", "2021-01-01", "2021-12-31")
    rows = [dict(model=name, **score(validation, weights)) for name, weights in WEIGHTS.items()]
    table = pd.DataFrame(rows).sort_values("rmse")
    reference = table.loc[table.model.eq("frozen_v4")].iloc[0]
    eligible = table.loc[(table.model.ne("frozen_v4")) &
                         (table.rmse <= .99 * reference.rmse) &
                         (table.mae <= reference.mae) &
                         (table.high_rmse <= 1.05 * reference.high_rmse)]
    selected = "frozen_v4" if eligible.empty else eligible.iloc[0].model
    print("2021 validation (fixed weights):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Selected on 2021:", selected)
    if selected == "frozen_v4":
        return
    history = read("test_2022_23.csv", "2022-01-01", "2023-12-31")
    comparison = []
    for period, part in (("2022-23", history), ("2022", history.loc["2022"]),
                         ("2023", history.loc["2023"])):
        for name in ("frozen_v4", selected):
            comparison.append(dict(period=period, model=name, **score(part, WEIGHTS[name])))
    print("2022-2023 historical comparison (not a new blind test):")
    print(pd.DataFrame(comparison).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("14-day block RMSE gain 95% interval:", gain_interval(history, WEIGHTS[selected]))


if __name__ == "__main__":
    main()
