"""Compare asymmetric and flow-change prediction intervals without retuning the point model."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import compare
from uncertainty import LEVEL, WINDOW, add_bounds, summarize


METHODS = ("flow_scaled", "signed", "directional", "change_scaled")
HISTORY_START = "2019-01-01"
SEED_DAY = "2018-12-31"
DEV_END = "2020-12-31"
TEST_START, TEST_END = "2022-01-01", "2023-12-31"


def empirical(values, probability):
    rank = min(len(values), max(1, math.ceil((len(values) + 1) * probability))) - 1
    return np.partition(values, rank)[rank]


def bounds(frame, method):
    if method == "flow_scaled":
        return add_bounds(frame, scaled=True)
    previous = frame.target.shift(1).to_numpy()
    point = frame.prediction.to_numpy()
    scale = (100 + previous) ** .75
    rising = point > previous
    if method == "change_scaled":
        scale += .25 * np.abs(point - previous)
    error = (frame.target.to_numpy() - point) / scale
    spans = np.full((len(frame), 2), np.nan)
    for i in range(WINDOW + 1, len(frame)):
        past = slice(i - WINDOW, i)
        residual = error[past]
        if method == "change_scaled":
            edge = empirical(np.abs(residual), LEVEL)
            spans[i] = (-edge, edge)
            continue
        lower = empirical(residual, (1 - LEVEL) / 2)
        upper = empirical(residual, (1 + LEVEL) / 2)
        if method == "directional":
            similar = residual[rising[past] == rising[i]]
            # Blend with the pooled estimates when enough comparable days exist.
            if len(similar) >= 60:
                lower = .5 * (lower + empirical(similar, (1 - LEVEL) / 2))
                upper = .5 * (upper + empirical(similar, (1 + LEVEL) / 2))
        spans[i] = (lower, upper)
    limits = np.maximum(point[:, None] + scale[:, None] * spans, 0)
    return pd.DataFrame(limits, columns=("lower", "upper"), index=frame.index)


def label_series():
    raw = pd.read_csv(compare.DATA / "glofas_daily_discharge.csv", parse_dates=["date"])
    return raw.loc[raw.station.eq("nanping") & raw.date.le(DEV_END)].set_index("date").river_discharge_m3s.sort_index()


def build_history(labels, include_outer):
    base = compare.read_period(compare.V4 / "validation_oof.csv", HISTORY_START, DEV_END)
    np.testing.assert_allclose(base.target, labels.loc[HISTORY_START:DEV_END], atol=1e-7, rtol=0)
    seed = pd.DataFrame({"target": [labels.loc[SEED_DAY]], "prediction": [np.nan]},
                        index=pd.DatetimeIndex([SEED_DAY], name="target_date"))
    parts = [seed, base]
    if include_outer:
        parts.append(compare.read_period(compare.V4 / "outer_2021.csv", "2021-01-01", "2021-12-31"))
    return pd.concat(parts)


def measure(history, year, threshold, method):
    forecast = history.join(bounds(history, method)).loc[year]
    if not np.isfinite(forecast).all().all() or (forecast.upper < forecast.lower).any():
        raise ValueError("Invalid or incomplete interval forecasts")
    altered = history.copy()
    cut = pd.Timestamp(f"{year}-07-01")
    altered.loc[cut:, "target"] += 500
    pd.testing.assert_frame_equal(bounds(history, method).loc[:cut], bounds(altered, method).loc[:cut])
    peak = summarize(forecast.loc[forecast.target >= threshold])
    return {"method": method, **summarize(forecast), "high_n": peak["n"], "high_covered": peak["covered"],
            "high_width": peak["mean_width"], "high_interval_score": peak["mean_interval_score"]}


def qualifies(row, base, margin):
    return (row["mean_interval_score"] <= (1 - margin) * base["mean_interval_score"] and
            row["covered"] >= base["covered"] - 2 and row["high_covered"] >= base["high_covered"])


def evaluate(output):
    labels = label_series()
    threshold = labels.loc["2014-02-01":"2019-12-31"].quantile(.95)
    dev = build_history(labels, include_outer=False)
    rows = [measure(dev, "2020", threshold, method) for method in METHODS]
    base = rows[0]
    for row in rows:
        row["eligible"] = row["method"] != base["method"] and qualifies(row, base, .03)
    candidates = [row for row in rows if row["eligible"]]
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "development_2020.csv", index=False)
    print(table.to_string(index=False))
    choice = dict(threshold_m3s=threshold, last_selection_label=DEV_END, selected=None, approved=False,
                  code_sha256=compare.fingerprint(Path(__file__)))
    if candidates:
        best = min(candidates, key=lambda r: r["mean_interval_score"])["method"]
        choice["selected"] = best
        outer = build_history(labels, include_outer=True)
        reference, candidate = [measure(outer, "2021", threshold, name) for name in (METHODS[0], best)]
        choice["approved"] = qualifies(candidate, reference, .01)
        table = pd.DataFrame([reference, candidate])
        table.to_csv(output / "validation_2021.csv", index=False)
        print(table.to_string(index=False))
    compare.write_json(output / "choice.json", choice)
    print("Decision:", choice)


def approved_method(output):
    choice = json.loads((output / "choice.json").read_text(encoding="utf-8"))
    if choice["code_sha256"] != compare.fingerprint(Path(__file__)):
        raise ValueError("Source changed since the development and validation decision")
    if not choice["approved"]:
        raise ValueError("No interval passed validation")
    return choice["selected"]


def test_history(labels):
    test = compare.read_period(compare.V4 / "test_2022_23.csv", TEST_START, TEST_END)
    return pd.concat([build_history(labels, include_outer=True), test])


def historical_test(output):
    selected = approved_method(output)
    labels = label_series()
    history = test_history(labels)
    threshold = labels.loc["2014-02-01":DEV_END].quantile(.95)
    # Preserve the high-flow definition already used in the published 37-day comparison.
    rows = []
    for method in (METHODS[0], selected):
        forecast = history.join(bounds(history, method)).loc[TEST_START:TEST_END]
        if not np.isfinite(forecast).all().all() or (forecast.upper < forecast.lower).any():
            raise ValueError("Invalid or incomplete test intervals")
        for period, part in (("2022-23", forecast), ("2022", forecast.loc["2022"]), ("2023", forecast.loc["2023"])):
            peak = summarize(part.loc[part.target >= threshold])
            rows.append(dict(method=method, period=period, **summarize(part), high_n=peak["n"],
                             high_covered=peak["covered"], high_width=peak["mean_width"],
                             high_interval_score=peak["mean_interval_score"]))
    report = pd.DataFrame(rows)
    report.to_csv(output / "historical_test_2022_23.csv", index=False)
    print(report.to_string(index=False))


def predict_next(output):
    from fast_point import stable_v4
    from threadpoolctl import threadpool_limits

    selected = approved_method(output)
    history = test_history(label_series())
    day = history.index[-1] + pd.Timedelta(days=1)
    with threadpool_limits(limits=2):
        point = stable_v4(compare.DATA, TEST_START, day, compare.V4).loc[day]
    history.loc[day, ["target", "prediction"]] = [np.nan, point]
    lower, upper = bounds(history, selected).loc[day]
    if not np.isfinite([point, lower, upper]).all() or upper < lower:
        raise ValueError("Missing data for next-day interval")
    print(f"{day.date()}  forecast={point:.2f}  {selected} 90% interval=[{lower:.2f}, {upper:.2f}] m3/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=compare.HERE / "interval_experiment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--historical-test", action="store_true")
    mode.add_argument("--next", action="store_true")
    args = parser.parse_args()
    if args.historical_test:
        historical_test(args.output)
    elif args.next:
        predict_next(args.output)
    else:
        evaluate(args.output)
