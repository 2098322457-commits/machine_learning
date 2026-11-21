"""Independently compare the old and new intervals on paired historical forecasts."""
import math

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import compare
import interval_experiment as experiment
from fast_point import stable_v4


def rank(values, probability):
    ordered = sorted(values)
    index = max(1, min(len(ordered), math.ceil((len(ordered) + 1) * probability))) - 1
    return ordered[index]


def reconstruct(history, dates, truth):
    point = history.prediction
    previous = truth.shift(1).loc[history.index]
    scale = (100 + previous) ** .75
    signed = (truth.loc[history.index] - point) / scale
    rising = point > previous
    rows = []
    for day in dates:
        lagged = slice(day - pd.Timedelta(days=365), day - pd.Timedelta(days=1))
        errors = signed.loc[lagged]
        if len(errors) != 365 or not np.isfinite(errors).all():
            raise ValueError(f"Incomplete calibration window for {day.date()}")
        old = rank(errors.abs(), .90)
        new = [rank(errors, .05), rank(errors, .95)]
        similar = errors[rising.loc[lagged].eq(rising[day])]
        if len(similar) >= 60:
            new = [(new[i] + rank(similar, tail)) / 2 for i, tail in enumerate((.05, .95))]
        p, s = point[day], scale[day]
        rows.append((max(0, p - s * old), p + s * old, max(0, p + s * new[0]), max(0, p + s * new[1])))
    return pd.DataFrame(rows, index=dates, columns=("old_low", "old_high", "new_low", "new_high"))


def losses(target, lower, upper):
    below, above = target < lower, target > upper
    width = upper - lower
    miss = (lower - target).clip(lower=0) + (target - upper).clip(lower=0)
    return pd.DataFrame(dict(covered=~(below | above), below=below, above=above,
                             width=width, score=width + 20 * miss))


def block_interval(values, block):
    rng = np.random.default_rng(2026)
    n = len(values)
    start = rng.integers(0, n, size=(4000, math.ceil(n / block), 1))
    index = ((start + np.arange(block)) % n).reshape(4000, -1)[:, :n]
    return np.quantile(values.to_numpy()[index].mean(axis=1), [.025, .975])


def main():
    method = experiment.approved_method(compare.HERE / "interval_experiment")
    if method != "directional":
        raise ValueError("This audit compares the approved directional interval against the previous interval")
    raw = pd.read_csv(compare.DATA / "glofas_daily_discharge.csv", parse_dates=["date"])
    truth = raw.loc[raw.station.eq("nanping")].set_index("date", verify_integrity=True).river_discharge_m3s
    history = experiment.test_history(experiment.label_series())
    dates = pd.date_range(experiment.TEST_START, experiment.TEST_END, name="target_date")
    target = truth.loc[dates]
    np.testing.assert_allclose(history.loc[dates, "target"], target, atol=1e-8, rtol=0)
    point = stable_v4(compare.DATA, dates[0], dates[-1])
    pd.testing.assert_index_equal(point.index, dates)
    np.testing.assert_allclose(point, history.loc[dates, "prediction"], atol=1e-6, rtol=0)
    print("Point RMSE and MAE, same for both interval methods:",
          np.sqrt(((point - target) ** 2).mean()), (point - target).abs().mean())
    independent = reconstruct(history, dates, truth)
    published = pd.read_csv(compare.RESULTS / "interval_predictions.csv", parse_dates=["target_date"])
    old = published.loc[published.method.eq("flow_scaled")].set_index("target_date").loc[dates]
    np.testing.assert_allclose(independent[["old_low", "old_high"]], old[["lower", "upper"]], atol=1e-7, rtol=0)
    np.testing.assert_allclose(independent[["new_low", "new_high"]],
                               experiment.bounds(history, method).loc[dates], atol=1e-7, rtol=0)
    print("Both intervals matched independent daily reconstruction.")
    cut = pd.Timestamp("2022-06-15")
    perturbed = history.copy()
    perturbed.loc[cut:, "target"] += 1000
    for rule in ("flow_scaled", method):
        pd.testing.assert_frame_equal(experiment.bounds(history, rule).loc[:cut],
                                     experiment.bounds(perturbed, rule).loc[:cut])
    print("Perturbing labels from 2022-06-15 did not change intervals through that date.")
    threshold = truth.loc["2014-02-01":"2020-12-31"].quantile(.95)
    peak = target.ge(threshold)
    data = {name: losses(target, independent[f"{name}_low"], independent[f"{name}_high"])
            for name in ("old", "new")}
    for period, mask in (("all", target.notna()), ("2022", dates.year == 2022),
                         ("2023", dates.year == 2023), ("high", peak)):
        for name, daily in data.items():
            subset = daily.loc[mask]
            print(period, name, f"covered={subset.covered.sum()}/{len(subset)}",
                  f"below={subset.below.sum()} above={subset.above.sum()}",
                  f"width={subset.width.mean():.6f} score={subset.score.mean():.6f}")
    for key in ("covered", "score"):
        old, new = (data[name][key].astype(float) for name in ("old", "new"))
        delta = (new - old) * 100 if key == "covered" else old - new
        print(f"Paired {key} gain (mean and 14-day block 95% range):",
              round(delta.mean(), 6), block_interval(delta, 14))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
