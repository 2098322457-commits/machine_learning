"""Rolling empirical intervals for the released point forecaster, using past errors only."""
import argparse
import json
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import compare

DAY = pd.Timedelta(days=1)
WINDOW = 365
LEVEL = .90


def release():
    lock = compare.read_lock()
    if lock["family"]:
        decision = json.loads((compare.RESULTS / "outer_decision.json").read_text(encoding="utf-8"))
        if decision["selection_sha256"] != compare.fingerprint(compare.RESULTS / "selection.json"):
            raise ValueError("Release differs from the validated model")
        if decision["accepted"]:
            return lock
    return {**lock, "family": None, "weight": 0.}


def archive(lock):
    start, stop = "2019-01-01", "2020-12-31"
    train = compare.read_period(compare.V4 / "validation_oof.csv", start, stop)
    # The 2021 release decision can first affect forecasts in January 2022.
    outer = compare.read_period(compare.V4 / "outer_2021.csv", "2021-01-01", "2021-12-31")
    source = compare.V4
    if lock["family"]:
        source = compare.RESULTS
        scores = json.loads((source / "test_metrics.json").read_text(encoding="utf-8"))
        if (scores["selection_sha256"] != compare.fingerprint(source / "selection.json") or
                scores["forecast_sha256"] != compare.fingerprint(source / "test_2022_23.csv")):
            raise ValueError("Published forecasts differ from the evaluated release")
    test = compare.read_period(source / "test_2022_23.csv", "2022-01-01", "2023-12-31")
    return pd.concat([train, outer, test])


def add_bounds(frame, scaled):
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max())):
        raise ValueError("Calibration requires contiguous, sorted forecast dates")
    # Target T's observable scale uses the value from T−1; T's error enters from T+1 onward.
    scale = (100 + frame.target.shift(1)) ** .75 if scaled else pd.Series(1., index=frame.index)
    normalized = (frame.prediction - frame.target).abs() / scale
    rank = math.ceil((WINDOW + 1) * LEVEL)
    quantile = normalized.shift(1).rolling(WINDOW).apply(lambda x: np.partition(x, rank-1)[rank-1], raw=True)
    half_width = scale * quantile
    return pd.DataFrame({"lower": (frame.prediction - half_width).clip(lower=0),
                         "upper": frame.prediction + half_width})


def summarize(part):
    if part.empty:
        raise ValueError("An evaluation period is empty")
    covered = part.target.between(part.lower, part.upper)
    alpha = 1 - LEVEL
    width = part.upper - part.lower
    penalty = ((part.lower - part.target).clip(lower=0) + (part.target - part.upper).clip(lower=0)) * (2 / alpha)
    return dict(n=len(part), covered=int(covered.sum()), coverage=covered.mean(), mean_width=width.mean(),
                mean_interval_score=(width + penalty).mean())


def evaluate():
    hist = archive(release())
    rows = []
    forecasts = []
    raw = pd.read_csv(compare.DATA / "glofas_daily_discharge.csv", parse_dates=["date"])
    flow = raw.loc[raw.station.eq("nanping")].set_index("date").river_discharge_m3s.sort_index()
    np.testing.assert_allclose(flow.loc[hist.index], hist.target, atol=1e-7, rtol=0)
    threshold = flow.loc["2014-02-01":"2020-12-31"].quantile(.95)
    for name, scaled in (("constant", False), ("flow_scaled", True)):
        bounds = add_bounds(hist, scaled)
        forecast = hist.join(bounds).loc["2021":]
        if not np.isfinite(forecast).all().all():
            raise ValueError("Incomplete interval forecast")
        # Check that future labels cannot affect today's already-issued interval.
        altered = hist.copy()
        cut = pd.Timestamp("2021-07-01")
        altered.loc[cut:, "target"] += 500
        pd.testing.assert_frame_equal(bounds.loc[:cut], add_bounds(altered, scaled).loc[:cut])
        for period, block in (("outer_2021", forecast.loc["2021"]), ("test_2022_23", forecast.loc["2022":])):
            for group, subset in (("all", block), ("high_flow", block[block.target >= threshold])):
                rows.append(dict(method=name, period=period, subset=group, **summarize(subset)))
        forecasts.append(forecast.assign(method=name))
    pd.concat(forecasts).to_csv(compare.RESULTS / "interval_predictions.csv")
    report = pd.DataFrame(rows)
    report.to_csv(compare.RESULTS / "interval_metrics.csv", index=False)
    print(report.to_string(index=False))


def predict_next():
    from fast_point import stable_v4
    lock = release()
    hist = archive(lock)
    next_day = hist.index[-1] + DAY
    # Match the anchored training dates of the published two-year test.
    start = "2022-01-01"
    base = stable_v4(compare.DATA, start, next_day, compare.V4)
    value = compare.blend(lock, compare.DATA, start, next_day, base).loc[next_day]
    hist.loc[next_day, ["target", "prediction"]] = [np.nan, value]
    bounds = add_bounds(hist, scaled=True).loc[next_day]
    if not np.isfinite([value, *bounds]).all():
        raise ValueError("Forecast or calibration history missing")
    print(f"{next_day.date()}  forecast={value:.2f}  empirical 90% interval=[{bounds.lower:.2f}, {bounds.upper:.2f}] m3/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next", action="store_true", help="Forecast the first unlabeled day following the archive")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        predict_next() if args.next else evaluate()
