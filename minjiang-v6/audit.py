"""Rebuild development forecasts from an archive physically ending in 2020."""
from importlib import import_module
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import compare


def historical_copy(destination):
    destination.mkdir()
    for source in compare.DATA.glob("*.csv"):
        if source.name == "stations.csv":
            shutil.copyfile(source, destination / source.name)
            continue
        with source.open("rb") as raw, (destination / source.name).open("wb") as filtered:
            header = raw.readline()
            if header.split(b",")[0].removeprefix(b"\xef\xbb\xbf") not in (b"date", b"time"):
                raise ValueError(f"Unrecognized date layout in {source}")
            filtered.write(header)
            filtered.writelines(line for line in raw if line[:10] <= b"2020-12-31")


def check_training():
    with TemporaryDirectory(prefix="minjiang-v6-audit-") as temp_name:
        root = Path(temp_name)
        data = root / "data"
        historical_copy(data)
        for family in compare.FAMILIES:
            model = import_module(family)
            temp_model = root / family
            model.validate(data, temp_model)
            start, end = "2018-01-01", "2020-12-31"
            expected = compare.read_period(compare.HERE / family / "oof.csv", start, end)
            rebuilt = compare.read_period(temp_model / "oof.csv", start, end)
            np.testing.assert_allclose(expected, rebuilt, rtol=0, atol=1e-6)
            # A target whose truth does not exist in the archive must still be forecastable.
            next_day = "2021-01-01"
            pred = model.predict(data, temp_model, next_day, next_day)
            if not pred.index.equals(pd.date_range(next_day, next_day)) or not np.isfinite(pred).all():
                raise ValueError(f"{family}: first unlabeled date not forecastable")
            print(f"{family}: {len(rebuilt)} dates reproduced from truncated tables; first unlabeled date forecastable.")


def check_intervals():
    from uncertainty import archive, release
    history = archive(release())
    raw = pd.read_csv(compare.RESULTS / "interval_predictions.csv", parse_dates=["target_date"])
    observed = raw.loc[raw.method.eq("flow_scaled")].set_index("target_date")
    for day in pd.to_datetime(["2021-01-01", "2022-01-01", "2023-12-31"]):
        history_before_day = history.loc[:day].tail(366)
        scales = (100 + history.target.shift(1)).loc[history_before_day.index] ** .75
        errors = (history_before_day.target - history_before_day.prediction).abs() / scales
        sorted_errors = sorted(errors.iloc[:-1])
        # ceil(366 * 0.90) is 330, hence element 329 in a zero-based sorted array.
        width = sorted_errors[329] * scales.iloc[-1]
        point = history_before_day.prediction.iloc[-1]
        np.testing.assert_allclose(observed.loc[day, ["lower", "upper"]].to_numpy(dtype=float),
                                   [max(0., point - width), point + width], atol=1e-7, rtol=0)
    summary = pd.read_csv(compare.RESULTS / "interval_metrics.csv")
    for row in summary.query("method == 'flow_scaled' and subset == 'all'").itertuples():
        period = observed.loc["2021"] if row.period == "outer_2021" else observed.loc["2022":]
        assert len(period) == row.n and period.target.between(period.lower, period.upper).sum() == row.covered
    print("Verified rolling-interval endpoints on three dates and coverage in both evaluation periods.")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        check_training()
        check_intervals()
