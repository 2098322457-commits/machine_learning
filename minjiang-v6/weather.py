"""Daily Nanping weather and flow transfer with stage-dependent coefficients."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "minjiang-original" / "data"
DAY = pd.Timedelta(days=1)
LAST = pd.Timestamp("2020-12-31")
BASINS = {"jianxi": ["pucheng", "wuyishan", "jianyang", "jianou"],
          "futun": ["guangze", "shaowu", "jiangle", "taining"],
          "shaxi": ["sanming", "yongan", "youxi"], "main": ["nanping"]}


def read_until(file, end, timestamp):
    """Only pass records through end to pandas (station-major and time-major files)."""
    cutoff = end.date().isoformat().encode("ascii")
    with Path(file).open("rb") as src:
        header = src.readline()
        if header.split(b",")[0].removeprefix(b"\xef\xbb\xbf").decode() != timestamp:
            raise ValueError(f"Unexpected timestamp column in {file}")
        contents = io.BytesIO(header + b"".join(line for line in src if line[:10] <= cutoff))
    return pd.read_csv(contents, parse_dates=[timestamp])


def load_features(data, end):
    data, end = Path(data), pd.Timestamp(end)
    coords = pd.read_csv(data/"stations.csv").set_index("station")
    raw = read_until(data/"glofas_daily_discharge.csv", end, "date")
    q = raw.pivot(index="date", columns="station", values="river_discharge_m3s").sort_index().asfreq("D")
    hourly = {}
    for variable in ["precipitation", "surface_pressure", "relative_humidity_2m", "temperature_2m", "wind_speed_10m"]:
        frame = read_until(data/f"openmeteo_hourly_{variable}.csv", end, "time").set_index("time")
        if frame.index.has_duplicates or set(frame.columns) != set(coords.index):
            raise ValueError(f"Duplicate timestamp or unknown weather station in {variable}")
        hourly[variable] = frame.sort_index().asfreq("h")
    nasa = read_until(data/"nasa_power_daily.csv", end, "date")
    nasa_rain = nasa.pivot(index="date", columns="station", values="precip_mm")
    nasa_temp = nasa.pivot(index="date", columns="station", values="temp_c")

    stage = q.nanping
    scale = 100 + .5 * (stage + stage.rolling(5).mean())
    driver, regime, weather = {}, {}, {}
    for site, flow in q.items():
        driver[f"{site}_flow"] = flow / scale
        for lag in range(5):
            driver[f"{site}_change{lag}"] = flow.diff().shift(lag) / scale
        driver[f"{site}_week"] = (flow - flow.rolling(7).mean()) / scale
        regime[f"{site}_rising"] = np.log((flow+10) / (flow.rolling(5).mean()+10))
        regime[f"{site}_share"] = np.log1p(flow / scale)
    regime["stage"] = np.log(scale/500)
    regime["slow_stage"] = np.log((100 + stage.rolling(14).mean())/scale)
    rain = hourly["precipitation"]
    # Each subtotal contains exactly six completed source-clock hourly bins.
    pulses = [rain[rain.index.hour // 6 == step].resample("D").sum(min_count=6) for step in range(4)]
    day_rain = rain.resample("D").sum(min_count=24)
    daily_meteo = {k: v.resample("D").mean() for k,v in hourly.items() if k != "precipitation"}
    evening = {k: v[v.index.hour>=18].resample("D").mean() for k,v in hourly.items() if k != "precipitation"}
    morning = {k: v[v.index.hour<6].resample("D").mean() for k,v in hourly.items() if k != "precipitation"}
    for basin, sites in BASINS.items():
        aggregate = day_rain[sites].mean(axis=1, skipna=False)
        wet = aggregate.rolling(14).mean()
        regime[basin+"_wet"] = np.log1p(wet)
        for lag in [0,1,2,3]:
            amt = aggregate.shift(lag) / scale
            driver[f"{basin}_rain{lag}"] = amt
            driver[f"{basin}_effective{lag}"] = amt * np.sqrt(1 + wet)
        for i, pulse in enumerate(pulses):
            driver[f"{basin}_pulse{i}"] = pulse[sites].mean(axis=1) / scale
        wet_pulse = np.log1p(pulses[-1][sites].mean(axis=1))
        weather[basin+"_nasa_rain"] = nasa_rain[sites].mean(axis=1)/scale
        weather[basin+"_nasa_disagreement"] = (nasa_rain[sites].mean(axis=1)-aggregate)/scale
        weather[basin+"_ground_temp_gap"] = nasa_temp[sites].mean(axis=1)-daily_meteo["temperature_2m"][sites].mean(axis=1)
        weather[basin+"_max24"] = day_rain[sites].max(axis=1)/scale
        weather[basin+"_frac_wet"] = (day_rain[sites]>1.).mean(axis=1)
        for name in daily_meteo:
            evening_mean = evening[name][sites].mean(axis=1)
            change = evening_mean - morning[name][sites].mean(axis=1)
            day_change = evening_mean.diff()
            for suffix, signal in [("shift",change),("change24",day_change)]:
                weather[f"{basin}_{name}_{suffix}"] = signal
                weather[f"{basin}_{name}_{suffix}_wet"] = signal * wet_pulse
            if name in ("relative_humidity_2m", "surface_pressure"):
                anomaly = evening_mean-evening_mean.shift(1).rolling(7).mean()
                weather[f"{basin}_{name}_anomaly"] = anomaly
                weather[f"{basin}_{name}_change_stage"] = day_change*regime["nanping_rising"]
    # Intensity-weighted movement of the rain field along two geographic axes.
    # Center offsets shrink to zero in dry bins; the wind CSV contains speed only.
    for axis in ("longitude", "latitude"):
        positions = coords.loc[rain.columns,axis] - coords.loc[rain.columns,axis].mean()
        positions /= positions.std()
        centers = [(p @ positions)/(p.sum(axis=1)+5) for p in pulses]
        weight = np.log1p(day_rain.mean(axis=1))
        for i, center in enumerate(centers):
            weather[f"{axis}_center{i}"] = center * weight
            if i:
                weather[f"{axis}_motion{i}"] = (center-centers[i-1]) * weight
    # Keep actual station-level pressure/humidity tendencies to distinguish
    # widespread frontal changes from changes confined to one tributary.
    for site in coords.index:
        weather[f"{site}_pressure_drop"] = evening["surface_pressure"][site].diff()
        weather[f"{site}_humidity_rise"] = evening["relative_humidity_2m"][site].diff()
        weather[f"{site}_rain6"] = pulses[-1][site]/scale
    return tuple(pd.DataFrame(block,index=q.index) for block in (driver,regime,weather)) + (scale, stage)


def standardized(block, fit, ev):
    scaler = StandardScaler().fit(block.loc[fit])
    return scaler.transform(block.loc[fit]), scaler.transform(block.loc[ev])


def fold(tables, fit, ev, choices):
    driver, regime, weather, scale, q = tables
    dt,de = standardized(driver,fit,ev)
    rt,re = standardized(regime,fit,ev)
    wt,we = standardized(weather,fit,ev)
    linear = 1 + dt@dt.T/dt.shape[1]
    linear_ev = 1 + de@dt.T/dt.shape[1]
    atm = wt@wt.T/wt.shape[1]
    atm_ev = we@wt.T/wt.shape[1]
    dist = cdist(rt,rt,"sqeuclidean")/rt.shape[1]
    dist_ev = cdist(re,rt,"sqeuclidean")/rt.shape[1]
    rel = scale[fit]/scale[fit].mean()
    correction = ((q.shift(-1)[fit]-q[fit])/scale[fit]).to_numpy()
    for bandwidth in sorted({c[0] for c in choices}):
        g = .25 + np.exp(-dist/bandwidth)
        ge = .25 + np.exp(-dist_ev/bandwidth)
        for weight in sorted({c[1] for c in choices if c[0]==bandwidth}):
            kernel = g*(linear+weight*atm)
            cross = ge*(linear_ev+weight*atm_ev)
            for b,w,penalty in choices:
                if (b,w) == (bandwidth,weight):
                    k = kernel.copy()
                    k.flat[::k.shape[0]+1] += penalty/rel.to_numpy()**2
                    coef = cho_solve(cho_factor(k, check_finite=False),correction,check_finite=False)
                    pred = (q[ev]+scale[ev]*(cross@coef)).clip(lower=0)
                    pred.index += DAY
                    yield (b,w,penalty),pred


def masks(tables):
    driver, regime, weather, scale, q = tables
    ok = np.isfinite(driver).all(axis=1) & np.isfinite(regime).all(axis=1) & np.isfinite(weather).all(axis=1)
    return ok & scale.notna() & q.notna(), driver.index + DAY


def validate(data, artifacts):
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    tables = load_features(data, LAST)
    ok, target_dates = masks(tables)
    options = list(itertools.product((1., 5., 20.), (.02, .15, .75), (.04, .4, 2.)))
    candidates = {choice: [] for choice in options}
    q = tables[-1]
    with threadpool_limits(limits=2):
        for year in (2018, 2019, 2020):
            train = ok & q.shift(-1).notna() & (target_dates >= "2014-02-01") & (target_dates.year < year)
            ev = ok & q.shift(-1).notna() & (target_dates.year==year)
            for choice, pred in fold(tables,train,ev,options):
                candidates[choice].append(pred)
    best = np.inf
    options_log = []
    grid = pd.date_range("2018-01-01", LAST, name="target_date")
    for choice, pieces in candidates.items():
        pred = pd.concat(pieces)
        if not pred.index.equals(grid) or not np.isfinite(pred).all():
            raise ValueError("Incomplete rolling OOF")
        rmse = float(np.sqrt(np.mean((pred-q.loc[pred.index])**2)))
        options_log.append(dict(params=choice,rmse=rmse))
        if rmse < best:
            best, best_choice, selected = rmse, choice, pred
    pd.DataFrame({"target":q.loc[grid],"prediction":selected},index=grid).to_csv(artifacts/"oof.csv")
    lock = dict(parameters=best_choice, last_selection_label=str(LAST.date()), score=best,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), grid=options_log)
    (artifacts/"choice.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    return lock


def predict(data, artifacts, start, stop):
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    lock = json.loads((Path(artifacts)/"choice.json").read_text(encoding="utf-8"))
    if start <= pd.Timestamp(lock["last_selection_label"]) or stop < start:
        raise ValueError("Use a forecast period strictly after selection")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != lock["source_sha256"]:
        raise ValueError("Weather model source differs from frozen selection")
    tables = load_features(data, stop-DAY)
    ok, dates = masks(tables)
    tr = ok & (dates >= "2014-02-01") & (dates < start) & tables[-1].shift(-1).notna()
    ev = ok & (dates >= start) & (dates <= stop)
    expected = pd.date_range(start,stop,name="target_date")
    if not dates[ev].equals(expected):
        raise ValueError("Missing required issue-day data")
    with threadpool_limits(limits=2):
        _, pred = next(fold(tables,tr,ev,[tuple(lock["parameters"])]))
    return pred.rename_axis("target_date").rename("prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--artifacts",type=Path,default=ROOT/"weather")
    args=parser.parse_args()
    lock=validate(args.data,args.artifacts)
    print("Weather model frozen through",lock["last_selection_label"],"parameters",lock["parameters"],"OOF RMSE",lock["score"])
