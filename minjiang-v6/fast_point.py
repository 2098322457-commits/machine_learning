"""Predict using the active branches of the frozen V4 model. No forecast results are cached."""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
V4_DIR = ROOT.parent/"minjiang-v4"
DATA = ROOT.parent/"minjiang-original"/"data"
DEFAULT_ARTIFACTS = V4_DIR/"results"


def load_v4():
    # The earlier projects have other modules named run.py. Bind V4 by its file path.
    spec = importlib.util.spec_from_file_location("minjiang_v4_reference", V4_DIR/"run.py")
    module = importlib.util.module_from_spec(spec)
    paths = sys.path.copy()
    try:
        sys.path.insert(0, str(V4_DIR))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = paths
    for dep in (module.prior, module.candidate, module.ar_model):
        if Path(dep.__file__).resolve().parent != V4_DIR:
            raise ImportError(f"Expected a V4 dependency, got {dep.__file__}")
    return module


V4 = load_v4()


def stable_v4(data, start, stop, artifacts=None):
    """Return the daily V4 predictions, preserving the original period fit and forecast dates.

    artifacts is the V4 results directory containing frozen.json; branch files live
    in its sibling directories. Each request reads its lock and checks all signatures.
    """
    artifacts = DEFAULT_ARTIFACTS if artifacts is None else Path(artifacts)
    model_root = artifacts.parent
    lock = json.loads((artifacts/"frozen.json").read_text(encoding="utf-8"))
    if lock["component_signatures"] != V4.signatures(model_root):
        raise ValueError("Model source or configuration changed since selection")
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if start not in (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")) or stop < start:
        raise ValueError("Use a locked benchmark start (2021-01-01 or 2022-01-01)")
    weights = pd.Series(lock["weights"], dtype=float).reindex(V4.KEYS)
    if set(lock["weights"]) != set(V4.KEYS) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid frozen weights")
    dates = pd.date_range(start, stop, name="target_date")
    pieces = {}
    for key, folder in zip(V4.KEYS, V4.SOURCE_DIRS, strict=True):
        if weights[key] == 0:
            continue
        args = (Path(data), model_root/folder, start, stop)
        if key == "prior":
            values = V4.prior.predict(*args)["prior"]
        elif key == "local":
            values = V4.candidate.locked_predict(*args)
        else:
            values = V4.ar_model.locked_predict(*args)
        if not values.index.equals(dates) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete predictions from {key}")
        pieces[key] = values
    active = pd.DataFrame(pieces, index=dates)
    return (active @ weights.loc[active.columns]).rename("prediction")


def benchmark(data, artifacts, repeats):
    if repeats < 3:
        raise ValueError("Run at least three timed calls per implementation")

    def original(first, last):
        return V4.frozen_predict(data, artifacts, first, last)["prediction"]

    def current(first, last):
        return stable_v4(data, first, last, artifacts)

    # Independent parity checks and one warm-up of the exact benchmark query.
    for first, last in [("2021-01-01","2021-12-31"), ("2022-01-01","2023-12-31"),
                        ("2022-01-01","2024-01-01")]:
        reference, optimized = original(first,last), current(first,last)
        pd.testing.assert_series_equal(reference, optimized, rtol=0, atol=1e-9)
        print(f"Checked {first}..{last}, {len(reference)} days, "
              f"max abs difference {float((reference-optimized).abs().max()):.12g}", flush=True)
    timers = {"original":[], "active":[]}
    versions = [("original", original), ("active", current)]
    for run in range(repeats):
        for name, implementation in versions if run % 2 == 0 else reversed(versions):
            tic = perf_counter()
            # Both calls read the locks and CSVs, refit, produce the exact same
            # Jan 2022..Jan 2024 forecast trajectory, then select the final scalar.
            result = float(implementation("2022-01-01","2024-01-01").iloc[-1])
            timers[name].append(perf_counter()-tic)
            print(f"Pair {run+1}, {name}: {timers[name][-1]:.6f} s; forecast {result:.12f}", flush=True)
    old, new = median(timers["original"]), median(timers["active"])
    print(f"Median: original {old:.6f} s, active {new:.6f} s, "
          f"speed ratio {old/new:.4f}, time reduction {100*(1-new/old):.2f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--date", help="Forecast one YYYY-MM-DD (2021 onwards)")
    action.add_argument("--benchmark", action="store_true", help="Verify parity and time both implementations")
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        if args.benchmark:
            benchmark(args.data, args.artifacts, args.repeats)
        else:
            day = pd.Timestamp(args.date)
            first = "2021-01-01" if day.year == 2021 else "2022-01-01"
            output = stable_v4(args.data, first, day, args.artifacts)
            print(output.iloc[[-1]].to_csv(), end="")


if __name__ == "__main__":
    main()
