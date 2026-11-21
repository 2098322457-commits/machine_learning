"""Independent time-key, score and future-label audit for the daily Nanping exercise."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SOURCE = "glofas_daily_discharge.csv"
FIRST_TARGET = pd.Timestamp("2014-02-01")
DAY = pd.Timedelta(days=1)


def require(ok, explanation):
    if not ok:
        raise ValueError(explanation)


def nanping(data):
    q = pd.read_csv(data/SOURCE, usecols=["date", "station", "river_discharge_m3s"], parse_dates=["date"])
    q = q.loc[q.station.eq("nanping")].set_index("date", verify_integrity=True).river_discharge_m3s
    require(q.index.equals(pd.date_range("2014-01-01", "2023-12-31")), "Unexpected Nanping date coverage")
    require(np.isfinite(q).all(), "Nonfinite GloFAS labels")
    return q


def read_predictions(path, first, last):
    df = pd.read_csv(path, index_col="target_date", parse_dates=["target_date"])
    dates = pd.date_range(first, last, freq="D")
    require(df.index.equals(dates), f"{path.name}: expected all {len(dates)} dates, in order, exactly once")
    return df


def compare_labels(frame, truth):
    np.testing.assert_allclose(frame.target, truth.reindex(frame.index), rtol=0, atol=1e-9,
                               err_msg="Exported targets differ from raw daily Nanping discharge")


def scores(actual, pred):
    if not pred.index.equals(actual.index):
        raise ValueError("Time keys differ")
    err = (pred - actual).to_numpy(dtype=float)
    require(len(err) and np.isfinite(err).all(), "Empty evaluation or invalid prediction")
    peak = (actual >= actual.quantile(0.95)).to_numpy()
    return dict(n=len(err), rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(abs(err))),
                peak_n=int(peak.sum()), peak_rmse=float(np.sqrt(np.mean(err[peak]**2))))


def print_score(label, row):
    print(f"{label}: n={row['n']} RMSE={row['rmse']:.6f} MAE={row['mae']:.6f} "
          f"high-flow n={row['peak_n']} RMSE={row['peak_rmse']:.6f}")


def audit_v3(archive, truth):
    old = read_predictions(archive/"feedback_predictions.csv", "2022-01-01", "2023-12-31")
    base = read_predictions(archive/"test_predictions.csv", "2022-01-01", "2023-12-31")
    compare_labels(base, truth)
    calendar = read_predictions(archive/"feedback_history_v3.csv", "2018-01-01", "2021-12-31")
    compare_labels(calendar, truth)
    record = pd.read_csv(archive/"final_metrics.csv", dtype={"period": str})
    record = record.set_index(["period", "model"], verify_integrity=True)
    for span in ("2022", "2023", "all"):
        block = old if span == "all" else old.loc[span]
        for name in block:
            measured = scores(truth.loc[block.index], block[name])
            stored = record.loc[(span, name)]
            np.testing.assert_allclose([measured[k] for k in stored.index], stored.to_numpy(),
                                       rtol=1e-10, atol=1e-9)
            if name == "v3_feedback":
                print_score(f"V3 {span}", measured)
    require(len(calendar.loc["2019":"2021"]) == 1096, "Unexpected feedback tuning period")
    frozen = json.loads((archive/"frozen.json").read_text(encoding="utf-8"))
    require(frozen["last_selection_date"] == "2021-12-31", "Unexpected V3 cutoff")


def audit_v4(out, truth):
    lock = json.loads((out/"frozen.json").read_text(encoding="utf-8"))
    require(lock["last_selection_label"] == "2020-12-31", "Model selection crossed inner/outer boundary")
    w = pd.Series(lock["weights"], dtype=float)
    require(set(w.index) == {"prior", "local", "ar"} and np.isfinite(w).all() and (w >= 0).all()
            and abs(w.sum()-1) < 1e-10, "Invalid frozen blend weights")
    record = pd.read_csv(out/"metrics.csv").set_index(["period", "model"], verify_integrity=True)
    audited = set()

    def compare_published(period, d):
        require(period in record.index.get_level_values("period"), f"No {period} metrics in the score file")
        np.testing.assert_allclose(d[w.index].to_numpy() @ w.to_numpy(), d.prediction, rtol=1e-9, atol=1e-7,
                                   err_msg=f"{period}: predictions differ from frozen blend")
        for name, saved in record.loc[period].iterrows():
            require(name in d, f"{period}: missing daily predictions for {name}")
            raw = scores(truth.loc[d.index], d[name])
            keys = ("n", "rmse", "mae", "peak_rmse")
            np.testing.assert_allclose([raw[k] for k in keys], saved[list(keys)].to_numpy(),
                                       rtol=1e-9, atol=1e-7, err_msg=f"{period}/{name}: metric mismatch")
            audited.add((period, name))
        require((period, "prediction") in audited, f"{period}: frozen model missing from report")

    inner = read_predictions(out/"validation_oof.csv", "2018-01-01", "2020-12-31")
    compare_labels(inner, truth)
    for name in ["local", "ar"]:
        print_score(f"V4 inner 2018-20 {name}", scores(truth.loc[inner.index], inner[name]))
    scored = inner.loc["2019":"2020"]
    require(len(scored) == 731, "Unexpected blend selection sample count")
    require(inner.loc["2018", ["prior", "prediction"]].isna().all().all(),
            "2018 should only initialize the error-history model")
    for name in ["prior", "prediction"]:
        print_score(f"V4 inner 2019-20 {name}", scores(truth.loc[scored.index], scored[name]))
    # Re-select from the 15 quarter-step weight combinations, using inner labels only.
    candidates = []
    cols = ["prior", "local", "ar"]
    for i in range(5):
        for j in range(5-i):
            weights = np.array([i, j, 4-i-j], dtype=float)/4
            residual = scored[cols].to_numpy()@weights - truth.loc[scored.index].to_numpy()
            candidates.append((float(np.sqrt(np.mean(residual**2))), np.count_nonzero(weights), weights))
    limit = min(row[0] for row in candidates) * 1.005
    selected = min((row for row in candidates if row[0] <= limit), key=lambda row: (row[1], row[0]))
    np.testing.assert_allclose(w[cols].to_numpy(), selected[2], atol=0, rtol=0)
    np.testing.assert_allclose(lock["inner_rmse"], selected[0], rtol=1e-9, atol=1e-8)
    require(lock["inner_n"] == 731, "Selection lock has wrong number of inner observations")
    compare_published("inner_2019_20", scored)
    periods = [("outer_2021.csv", "2021-01-01", "2021-12-31"),
               ("test_2022_23.csv", "2022-01-01", "2023-12-31")]
    for file, start, stop in periods:
        d = read_predictions(out/file, start, stop)
        compare_labels(d, truth)
        print_score(f"V4 {start}..{stop}", scores(truth.loc[d.index], d.prediction))
        compare_published(Path(file).stem, d)
        if start == "2022-01-01":
            for year in ["2022", "2023"]:
                rows = d.loc[year]
                print_score(f"V4 test {year}", scores(truth.loc[rows.index], rows.prediction))
                compare_published("test_"+year, rows)
    require(audited == set(record.index), "Score file contains an unverified period")


def audit_perturbation(data, out):
    # The preserved V3 feedback module imports its own "run" module internally.
    # Load V4 under a distinct name so that V3's module name remains available.
    spec = importlib.util.spec_from_file_location("minjiang_v4_driver", ROOT/"run.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    frozen_predict = driver.frozen_predict

    # Keep the original fit interval and change every Nanping label from this day.
    cutoff = pd.Timestamp("2021-07-01")
    first, last = "2021-01-01", "2021-12-31"
    stored = read_predictions(out/"outer_2021.csv", first, last).prediction
    clean = frozen_predict(data, out, first, last).prediction
    require(clean.index.equals(stored.index), "Forecast entry point changed the evaluation dates")
    np.testing.assert_allclose(clean, stored, rtol=1e-8, atol=1e-7)
    with TemporaryDirectory(prefix="nanping-v4-") as name:
        alt = Path(name)
        for f in data.glob("*.csv"):
            if f.name != SOURCE:
                shutil.copyfile(f, alt/f.name)
        q = pd.read_csv(data/SOURCE, parse_dates=["date"])
        future = (q.date >= cutoff) & q.station.eq("nanping")
        require(future.any(), "Perturbation window empty")
        q.loc[future, "river_discharge_m3s"] *= 1.15
        q.loc[future, "river_discharge_m3s"] += 50
        q.to_csv(alt/SOURCE, index=False)
        changed = frozen_predict(alt, out, first, last).prediction
    require(changed.index.equals(stored.index) and np.isfinite(changed).all(), "Invalid perturbed forecast")
    np.testing.assert_allclose(clean.loc[:cutoff], changed.loc[:cutoff], rtol=1e-8, atol=1e-7,
                               err_msg="Prediction at/before first perturbed target depends on future labels")
    late_change = (changed.loc[cutoff+DAY:] - clean.loc[cutoff+DAY:]).abs().max()
    require(late_change > 1e-6, "Perturbation failed to reach subsequent flow-based forecasts")
    print(f"Perturbation passed. Through {cutoff.date()} predictions match; subsequent "
          f"maximum absolute change {late_change:.6f} m3/s.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT.parent/"minjiang-original"/"data")
    p.add_argument("--v3", type=Path, default=ROOT.parent/"minjiang-v3")
    p.add_argument("--out", type=Path, default=ROOT/"results")
    p.add_argument("--legacy-only", action="store_true")
    args = p.parse_args()
    truth = nanping(args.data)
    days = truth.index[truth.index >= FIRST_TARGET]
    for year in range(2018, 2023):
        print(f"Before {year}: {sum(days.year < year)} eligible daily training labels")
    audit_v3(args.v3, truth)
    if not args.legacy_only:
        audit_v4(args.out, truth)
        audit_perturbation(args.data, args.out)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
