"""Compare the three forecast families under the fixed chronological protocol."""
import argparse
from importlib import import_module
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "minjiang-v4" / "results"
DATA = HERE.parent / "minjiang-original" / "data"
FAMILIES = ("transfer", "weather", "trees")
RESULTS = HERE / "results"


def read_period(path, start, stop):
    frame = pd.read_csv(path, index_col="target_date", parse_dates=True).loc[start:stop]
    if not frame.index.equals(pd.date_range(start, stop)):
        raise ValueError(f"Missing, unordered or repeated target dates in {path}")
    if not np.isfinite(frame[["prediction", "target"]]).all().all():
        raise ValueError(f"Missing predictions or targets in {path}")
    return frame[["prediction", "target"]]


def score(target, prediction):
    if not prediction.index.equals(target.index) or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction or target date mismatch")
    error = prediction - target
    peak = target >= target.quantile(.95)
    return dict(rmse=float(np.sqrt((error ** 2).mean())), mae=float(error.abs().mean()),
                peak_rmse=float(np.sqrt((error[peak] ** 2).mean())))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    paths = [HERE / "protocol.md", HERE / "compare.py", V4 / "frozen.json", V4 / "validation_oof.csv"]
    for family in FAMILIES:
        paths += [HERE / f"{family}.py", HERE / family / "choice.json", HERE / family / "oof.csv"]
    return {p.relative_to(HERE.parent).as_posix(): fingerprint(p) for p in paths}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    start, stop = "2019-01-01", "2020-12-31"
    base = read_period(V4 / "validation_oof.csv", start, stop)
    base_score = score(base.target, base.prediction)
    years = (2019, 2020)
    reference = {year: score(base.target.loc[str(year)], base.prediction.loc[str(year)]) for year in years}
    rows = []
    for family in FAMILIES:
        contender = read_period(HERE / family / "oof.csv", start, stop)
        np.testing.assert_allclose(base.target, contender.target, atol=1e-7, rtol=0)
        for weight in (.25, .5, .75, 1.):
            prediction = (1 - weight) * base.prediction + weight * contender.prediction
            scores = score(base.target, prediction)
            yearly = {k: score(base.target.loc[str(k)], prediction.loc[str(k)])["rmse"] for k in years}
            passed = (scores["rmse"] <= .98 * base_score["rmse"] and
                      scores["mae"] <= 1.03 * base_score["mae"] and
                      all(yearly[k] <= reference[k]["rmse"] for k in years))
            rows.append(dict(family=family, weight=weight, **scores, rmse_2019=yearly[2019],
                             rmse_2020=yearly[2020], passed=passed))
    table = pd.DataFrame(rows).sort_values(["rmse", "family", "weight"])
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "development.csv", index=False)
    eligible = table[table.passed]
    family, weight = (None, 0.) if eligible.empty else (eligible.iloc[0].family, float(eligible.iloc[0].weight))
    lock = dict(last_selection_label=stop, n=len(base), family=family, weight=weight, reference=base_score,
                reference_yearly=reference, signatures=dependencies())
    write_json(RESULTS / "selection.json", lock)
    print(table.to_string(index=False))
    print("Frozen candidate:", family, weight)


def read_lock():
    lock = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    if lock["signatures"] != dependencies():
        raise ValueError("Frozen selection differs from current files. Repeat development stage.")
    return lock


def blend(lock, data, start, stop, base):
    family = lock["family"]
    if family is None:
        return base.copy()
    candidate = import_module(family).predict(data, HERE / family, start, stop)
    if not candidate.index.equals(base.index) or not np.isfinite(candidate).all():
        raise ValueError("Candidate returned invalid or misaligned forecasts")
    return (1 - lock["weight"]) * base + lock["weight"] * candidate


def evaluate(data, test):
    lock = read_lock()
    if not lock["family"]:
        print("Development did not qualify a candidate. Retaining V4.")
        return
    if test:
        decision = json.loads((RESULTS / "outer_decision.json").read_text(encoding="utf-8"))
        if decision["selection_sha256"] != fingerprint(RESULTS / "selection.json"):
            raise ValueError("Outer decision differs from the current lock")
        if not decision["accepted"]:
            print("Outer validation did not qualify the candidate. Retaining V4.")
            return
    period, start, stop = (("test_2022_23", "2022-01-01", "2023-12-31") if test else
                           ("outer_2021", "2021-01-01", "2021-12-31"))
    base = read_period(V4 / f"{period}.csv", start, stop)
    prediction = blend(lock, data, start, stop, base.prediction)
    ref, new = score(base.target, base.prediction), score(base.target, prediction)
    forecast_file = RESULTS / f"{period}.csv"
    base.rename(columns={"prediction":"v4"}).assign(prediction=prediction).to_csv(forecast_file)
    record = dict(period=period, n=len(base), reference=ref, contender=new,
                  selection_sha256=fingerprint(RESULTS / "selection.json"), forecast_sha256=fingerprint(forecast_file))
    if not test:
        record["accepted"] = (new["rmse"] <= .99 * ref["rmse"] and new["mae"] <= 1.03 * ref["mae"] and
                              new["peak_rmse"] <= 1.05 * ref["peak_rmse"])
    write_json(RESULTS / ("test_metrics.json" if test else "outer_decision.json"), record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "validate", "test", "all"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage in ("select", "all"):
            freeze()
        if args.stage in ("validate", "all"):
            evaluate(DATA, test=False)
        if args.stage in ("test", "all"):
            evaluate(DATA, test=True)
