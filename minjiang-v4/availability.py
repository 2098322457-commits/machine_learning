"""Score the frozen local regression with on-time and one-day-late observations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import candidate


def compare(data, choices, output):
    flow = pd.read_csv(data / "glofas_daily_discharge.csv", parse_dates=["date"])
    target = flow.loc[flow.station.eq("nanping")].set_index("date").river_discharge_m3s
    target = target.sort_index().asfreq("D")
    rows = []
    # On Jan 1 2021 the delayed scenario could not have used the Dec 31 selection label.
    for period, start, end in (
        ("validation", "2021-01-02", "2021-12-31"),
        ("test", "2022-01-01", "2023-12-31"),
    ):
        dates = pd.date_range(start, end)
        truth = target.loc[dates]
        for delay in (0, 1):
            local = candidate.locked_predict(data, choices, start, end, delay_days=delay)
            forecasts = {
                "local": local,
                "persistence": target.shift(1 + delay).loc[dates],
            }
            for name, forecast in forecasts.items():
                if not forecast.index.equals(dates):
                    raise ValueError("Forecasts and targets have different dates")
                error = forecast.to_numpy() - truth.to_numpy()
                if not np.isfinite(error).all():
                    raise ValueError("Incomplete forecasts or targets")
                rows.append(dict(period=period, start=start, end=end, n=len(dates),
                                 latest_source_lag_days=1 + delay, model=name,
                                 rmse=np.sqrt(np.mean(error ** 2)), mae=np.mean(np.abs(error))))
    report = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=candidate.DATA)
    parser.add_argument("--choices", type=Path, default=candidate.ROOT / "candidate_results")
    parser.add_argument("--output", type=Path, default=candidate.ROOT / "results" / "availability.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        print(compare(args.data, args.choices, args.output).to_string(index=False))
