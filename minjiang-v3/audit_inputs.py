"""Check the eight group CSVs and test forecast causality against altered raw files."""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
FLOW = "glofas_daily_discharge.csv"
NASA = "nasa_power_daily.csv"
HOURS = {
    "precipitation": (0, None),
    "relative_humidity_2m": (0, 100),
    "surface_pressure": (0, None),
    "temperature_2m": (None, None),
    "wind_speed_10m": (0, None),
}
DAYS = pd.date_range("2014-01-01", "2023-12-31", freq="D")
HOURLY = pd.date_range(DAYS[0], DAYS[-1] + pd.Timedelta(hours=23), freq="h")
FLOW_STATIONS = {"nanping", "jianou", "shaowu", "sanming", "shuikou_dam"}
DAY = pd.Timedelta(days=1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_values(values, label, lower=None, upper=None):
    numbers = values.to_numpy(dtype=float)
    require(np.isfinite(numbers).all(), f"{label}: missing or infinite value")
    if lower is not None:
        require(numbers.min() >= lower, f"{label}: below {lower}")
    if upper is not None:
        require(numbers.max() <= upper, f"{label}: above {upper}")


def check_tables(data):
    sites = pd.read_csv(data/"stations.csv")
    require(list(sites) == ["station", "station_cn", "latitude", "longitude", "elevation_m"],
            "Unexpected station metadata columns")
    require(len(sites) == 12 and sites.station.is_unique and not sites.isna().any().any(),
            "Expected 12 unique complete weather stations")
    check_values(sites.latitude, "latitude", -90, 90)
    check_values(sites.longitude, "longitude", -180, 180)
    check_values(sites.elevation_m, "elevation")
    names = set(sites.station)
    readouts = []
    for variable, (lower, upper) in HOURS.items():
        filename = "openmeteo_hourly_" + variable + ".csv"
        df = pd.read_csv(data/filename, index_col="time", parse_dates=["time"])
        require(df.index.equals(HOURLY), f"{filename}: missing, duplicate or unordered hours")
        require(set(df.columns) == names, f"{filename}: station columns differ from metadata")
        check_values(df, filename, lower, upper)
        readouts.append((filename, len(df), len(df.columns)))
    for filename, stations, value_cols in [(NASA, names, ["precip_mm", "temp_c"]),
                                           (FLOW, FLOW_STATIONS, ["river_discharge_m3s"])]:
        df = pd.read_csv(data/filename, parse_dates=["date"])
        require(list(df.columns) == ["date", "station", *value_cols], f"{filename}: unexpected columns")
        require(set(df.station) == stations, f"{filename}: unexpected stations")
        for station, rows in df.groupby("station"):
            require(pd.DatetimeIndex(rows.date).equals(DAYS), f"{filename}: bad calendar at {station}")
        for col in value_cols:
            check_values(df[col], col, None if col == "temp_c" else 0)
        readouts.append((filename, len(df), len(stations)))
    for filename, rows, nsites in readouts:
        print(f"{filename}: {rows:,} time rows; {nsites} stations")
    print("stations.csv: 12 weather stations. All eight CSV checks passed.")


def change_raw_data(data, dest, cutoff):
    """Alter everything dated on/after cutoff; preserve the actual source files."""
    shutil.copyfile(data/"stations.csv", dest/"stations.csv")
    for filename, date_col, keys in [
        (FLOW, "date", ["river_discharge_m3s"]), (NASA, "date", ["precip_mm", "temp_c"]),
        *[("openmeteo_hourly_" + v + ".csv", "time", None) for v in HOURS],
    ]:
        frame = pd.read_csv(data/filename)
        future = pd.to_datetime(frame[date_col]) >= cutoff
        columns = frame.columns.difference([date_col]) if keys is None else keys
        require(future.any() and not future.all(), f"Invalid cutoff for {filename}")
        for col in columns:
            values = frame[col].astype(float)
            if "humidity" in filename:
                values.loc[future] = 5 + 0.9 * values.loc[future]
            else:
                values.loc[future] = 1.1 * values.loc[future] + 0.5
            frame[col] = values
        frame.to_csv(dest/filename, index=False)


def equal_early_changed_later(before, after, cutoff, label, expect_later=True):
    pd.testing.assert_index_equal(before.index, after.index)
    early = before.index <= cutoff
    require(early.any() and (~early).any(), f"{label}: no observations straddling cutoff")
    if isinstance(before, pd.DataFrame):
        pd.testing.assert_index_equal(before.columns, after.columns)
    np.testing.assert_allclose(before.loc[early], after.loc[early], rtol=1e-12, atol=1e-10,
                               equal_nan=True, err_msg=f"{label}: future input changed earlier output")
    if expect_later:
        same = np.isclose(before.loc[~early], after.loc[~early], equal_nan=True, rtol=1e-9, atol=1e-8)
        require(not same.all(), f"{label}: later observations should respond to perturbation")


def verify_scale(data, altered, cutoff, stop, policy):
    from kernel_routing import locked_predict as kernel_predict
    from scale_routing import design, locked_predict as scale_predict

    previous, scale, current_flow = design(data, stop - DAY)
    changed, changed_scale, changed_flow = design(altered, stop - DAY)
    # design() uses the just-completed source day as the index.
    for before, after, name in [(previous, changed, "scale features"), (scale, changed_scale, "scale"),
                                (current_flow, changed_flow, "previous daily flow")]:
        equal_early_changed_later(before, after, cutoff - DAY, name)
    # Cross-check that retained windows can also be rebuilt using a truncated raw horizon.
    shortened, _, _ = design(data, cutoff - DAY)
    pd.testing.assert_frame_equal(previous.loc[:cutoff-DAY], shortened)
    start = cutoff - pd.Timedelta(days=32)
    expected = pd.date_range(start, stop, freq="D")
    for name, predict in [("scale", scale_predict), ("kernel", kernel_predict)]:
        for delay in (0, 1):
            reference = predict(data, policy, start, stop, delay_days=delay)
            perturbed = predict(altered, policy, start, stop, delay_days=delay)
            require(reference.index.equals(expected), f"{name} predictor omitted requested target dates")
            equal_early_changed_later(reference, perturbed, cutoff + delay*DAY, f"{name} forecast")
    print("Scale and kernel routing: prefix rebuild and prediction perturbation passed "
          "at normal availability and with one extra day of delay.")


def verify_routing(data, altered, cutoff, stop, lock_file):
    from routing import build_tables, fit_predict, locked_predict

    base, y = build_tables(data, stop)
    mutated, y2 = build_tables(altered, stop)
    prefix, _ = build_tables(data, cutoff)
    raw = pd.read_csv(data/FLOW, parse_dates=["date"]).pivot(index="date", columns="station",
                                                           values="river_discharge_m3s")
    np.testing.assert_allclose(y, raw.nanping.loc[y.index], rtol=0, atol=0)
    require(y2.loc[cutoff] != y.loc[cutoff], "Target on the cutoff day did not change")
    for site in FLOW_STATIONS:
        prior = raw[site].reindex(y.index-DAY)
        np.testing.assert_allclose(base["group"][f"q_{site}_lag0"], prior, rtol=0, atol=0)
    for name, x in base.items():
        equal_early_changed_later(x, mutated[name], cutoff, name)
        pd.testing.assert_frame_equal(prefix[name], x.loc[:cutoff])
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    family = lock["family"]
    dates = y.index
    fit = dates < cutoff - pd.Timedelta(days=32)
    pred = ~fit
    # Both runs see identical calibration data. Only the post-cutoff inputs differ.
    def forecast(x, target):
        vals = fit_predict(family, lock["alpha"], x.loc[fit], target.loc[fit], x.loc[pred])
        return pd.Series(vals, index=dates[pred])
    before = forecast(base[family], y)
    after = forecast(mutated[family], y2)
    equal_early_changed_later(before, after, cutoff, "routing forecast")
    unlabeled_day = DAYS[-1] + DAY
    _, labels = build_tables(data, unlabeled_day, require_labels=False)
    require(unlabeled_day in labels.index and pd.isna(labels.loc[unlabeled_day]),
            "Expected a genuinely unlabeled date after the end of the raw horizon")
    next_forecast = locked_predict(data, lock_file.parent, unlabeled_day, unlabeled_day)
    require(next_forecast.index.equals(pd.DatetimeIndex([unlabeled_day])) and np.isfinite(next_forecast.iloc[0]),
            "Unlabeled target was not predicted")
    print(f"Routing: {len(base)} feature families, independent lag/label check, "
          f"prefix rebuild, prediction perturbation and unlabeled {unlabeled_day.date()} passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent/"minjiang-original"/"data")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--policy", type=Path, default=ROOT/"experiments")
    parser.add_argument("--routing-lock", type=Path, default=ROOT/"routing_results"/"lock.json")
    args = parser.parse_args()
    check_tables(args.data)
    if args.tables_only:
        return
    cutoff = pd.Timestamp("2018-07-01")
    stop = cutoff + pd.Timedelta(days=12)
    with TemporaryDirectory(prefix="minjiang-causal-") as folder:
        changed = Path(folder)
        change_raw_data(args.data, changed, cutoff)
        verify_scale(args.data, changed, cutoff, stop, args.policy)
        verify_routing(args.data, changed, cutoff, stop, args.routing_lock)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
