"""Next-day Nanping discharge. All input values are read from the group's data directory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "minjiang-original" / "data"
BASINS = {"jianxi":["jianyang","jianou","pucheng","wuyishan"],
          "futunxi":["shaowu","guangze","taining","jiangle"],
          "shaxi":["sanming","yongan","youxi"], "ganliu":["nanping"]}
SITES = ["nanping","jianou","shaowu","sanming","shuikou_dam"]
DAY = pd.Timedelta(days=1)


def load_hours(data, variable, stop):
    f = pd.read_csv(data / f"openmeteo_hourly_{variable}.csv", index_col="time", parse_dates=["time"])
    if f.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in {variable}")
    return f.sort_index().loc[:stop].asfreq("h")


def group_and_v2(q, rain, weather):
    # Reconstruct the two previous feature designs from the raw CSV files.
    totals = rain.resample("D").sum(min_count=24)
    series = {basin:totals[names].mean(axis=1) for basin,names in BASINS.items()}
    series["basin"] = totals.mean(axis=1)
    base = {}
    for basin,s in series.items():
        name = "rain_"+basin
        base[name] = s
        for lag in range(1,8):
            base[f"{name}_lag{lag}"] = s.shift(lag)
        for w in (3,7,30):
            base[f"{name}_sum{w}"] = s.rolling(w).sum()
            base[f"{name}_sum{w}_lag1"] = s.rolling(w).sum().shift(1)
    for var,s in weather.items():
        mean = s.resample("D").mean()
        base[f"met_{var}"] = mean
        base[f"met_{var}_lag1"] = mean.shift(1)
    for site in sorted(q):
        for lag in range(4):
            base[f"q_{site}_lag{lag}"] = q[site].shift(lag)
    x = pd.DataFrame(base)
    x["doy_sin"] = np.sin(2*np.pi*x.index.dayofyear/365.25)
    x["doy_cos"] = np.cos(2*np.pi*x.index.dayofyear/365.25)
    x["month"] = x.index.month
    x["is_flood_season"] = x.index.month.isin(range(4,10)).astype(int)
    extra = {}
    for site in SITES:
        s = q[site]
        for lag in range(4):
            extra[f"{site}_sqrt{lag}"] = np.sqrt(s.shift(lag))
            extra[f"{site}_pow{lag}"] = s.shift(lag)**1.5
        for lag in (1,2):
            delta = s.diff(lag)
            extra[f"{site}_rising{lag}"] = delta.clip(lower=0)
            extra[f"{site}_transport{lag}"] = np.sqrt(s)*delta
    for basin in BASINS:
        for lag in range(3):
            r = series[basin].shift(lag)
            extra[f"{basin}_storm{lag}"] = r**1.5
            extra[f"{basin}_wet{lag}"] = r*np.sqrt(x[f"rain_{basin}_sum7_lag1"])
            extra[f"{basin}_state{lag}"] = r*np.sqrt(q.nanping)
    return x, pd.concat([x,pd.DataFrame(extra)],axis=1)


def channel_features(q):
    extra = {}
    # Routing coefficients can vary with current depth. Fixed breakpoints use
    # no sample quantiles and therefore do not depend on future river records.
    total = q.nanping
    gates = {str(k):total/(k+total) for k in (150,800,2500)}
    for site,s in q.items():
        delta = s.diff()
        for lag in range(1,6):
            extra[f"channel_{site}_dq{lag}"] = delta.shift(lag)
        for label,gate in gates.items():
            for name,flux in {"level":s,"dq":delta,"prev":delta.shift(1),
                              "previous2":delta.shift(2),"rise":delta.clip(lower=0)}.items():
                extra[f"channel_{site}_{label}_{name}"] = flux*gate
        # Measures curvature of the recent hydrograph and direction of routing.
        for lag in (0,1):
            relative = delta.shift(lag) / (50 + s.shift(lag) + s.shift(lag+1))
            extra[f"{site}_relative{lag}"] = relative*s
            extra[f"{site}_sharp{lag}"] = relative.abs()*delta.shift(lag)
        ratio = np.log((50+s)/(50+total))
        extra[f"channel_{site}_ratio"] = ratio*s
        extra[f"channel_{site}_ratio_dq"] = ratio*delta
        extra[f"channel_{site}_ratio_dn"] = ratio*total.diff()
    branches = q.jianou + q.sanming + q.shaowu
    spread = branches - total
    downstream = q.shuikou_dam - total
    for label,s in {"up":spread,"down":downstream}.items():
        for lag in (0,1,2):
            gap = s.shift(lag)
            extra[f"gap_{label}_{lag}_positive"] = gap.clip(lower=0)
            extra[f"gap_{label}_{lag}_signed"] = gap*gap.abs()/(100+total)
    return pd.DataFrame(extra)


def weather_features(rain, weather, nasa, q):
    extra = {}
    nasa_daily = nasa.pivot(index="date",columns="station",values="precip_mm").asfreq("D")
    for basin,names in BASINS.items():
        h = rain[names].mean(axis=1)
        daily = h.resample("D").sum(min_count=24)
        r_nasa = nasa_daily[names].mean(axis=1)
        extra[f"{basin}_nasa_rain"] = r_nasa
        extra[f"{basin}_nasa_prev"] = r_nasa.shift(1)
        wetness = daily.ewm(halflife=7,adjust=False).mean().shift(1)
        # Resolve storm timing and spatial coverage before the issue cutoff.
        for start in (0,6,12,18):
            pulse = h[h.index.hour.isin(range(start,start+6))].resample("D").sum(min_count=6)
            for lag in (0,1):
                extra[f"{basin}_hour{start}_lag{lag}"] = pulse.shift(lag)
            extra[f"{basin}_hour{start}_wet"] = pulse*np.sqrt(wetness)
        for name in names:
            site_rain = rain[name].resample("D").sum(min_count=24)
            extra[f"rain_{name}_deviation"] = site_rain - daily
            extra[f"rain_{name}_devlag"] = (site_rain-daily).shift(1)
        extra[f"{basin}_max3h"] = h.rolling(3).sum().resample("D").max()
        for span in (2,7,20):
            storage = daily.ewm(halflife=span,adjust=False).mean().shift(1)
            extra[f"{basin}_soil{span}"] = daily * storage/(5+storage)
        extra[f"{basin}_storage"] = daily * q.nanping/(q.nanping+500)
    for name,s in weather.items():
        recent = s[s.index.hour>=18].resample("D").mean()
        early = s[s.index.hour<6].resample("D").mean()
        extra[f"weather_{name}_trend"] = recent-early
    return pd.DataFrame(extra)


def network_features(q):
    # First-order change in travel time depends on hydrograph shape and on
    # which tributary supplies the water. Modulators are bounded near low flow.
    state = {}
    for site in SITES:
        s = q[site]
        state[site+"_share"] = s/(q.nanping+s+100)
        state[site+"_change"] = s.diff()/(s+s.shift(1)+50)
    fluxes = {}
    for site in SITES:
        s = q[site]
        fluxes[site+"_level"] = s
        fluxes[site+"_change"] = s.diff()
        fluxes[site+"_prev"] = s.diff().shift(1)
    return pd.DataFrame({f"network_{a}_{b}":gate*flux for a,gate in state.items()
                         for b,flux in fluxes.items()})


def build_tables(data, stop, require_labels=True):
    """Return model inputs and targets indexed by target date up to stop (inclusive)."""
    data = Path(data)
    target_stop = pd.Timestamp(stop)
    q = pd.read_csv(data/"glofas_daily_discharge.csv",parse_dates=["date"])
    q = q.pivot(index="date",columns="station",values="river_discharge_m3s").sort_index().loc[:target_stop].asfreq("D")
    issue_end = target_stop - pd.Timedelta(seconds=1)
    rain = load_hours(data,"precipitation",issue_end)
    weather = {var.split("_")[0]:load_hours(data,var,issue_end).mean(axis=1) for var in
               ("temperature_2m","relative_humidity_2m","surface_pressure","wind_speed_10m")}
    nasa = pd.read_csv(data/"nasa_power_daily.csv",parse_dates=["date"])
    nasa = nasa[nasa.date < target_stop]
    flows = q.loc[:issue_end]
    group,v2 = group_and_v2(flows,rain,weather)
    channels = channel_features(flows)
    storms = weather_features(rain,weather,nasa,flows)
    network = network_features(flows)
    valid = group.dropna().index
    y = q.nanping.reindex(valid+DAY).rename("y")
    families = {"group":group,"v2":v2,
                "channel":pd.concat([v2,channels],axis=1),
                "storm":pd.concat([v2,storms],axis=1),
                "channel_storm":pd.concat([v2,channels,storms],axis=1),
                "network":pd.concat([v2,channels,network],axis=1),
                "network_storm":pd.concat([v2,channels,network,storms],axis=1)}
    for key,x in families.items():
        x = x.loc[valid].set_axis(y.index)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError(f"Nonfinite covariates in {key}; check hourly coverage and common dates")
        families[key] = x
    if require_labels and y.isna().any():
        raise ValueError("Missing Nanping next-day target")
    return families,y


def fit_predict(family, penalty, x_train, y_train, x_pred):
    model = make_pipeline(StandardScaler(), Ridge(alpha=penalty,solver="cholesky"))
    if family in ("group","v2"):
        return model.fit(x_train,y_train).predict(x_pred).clip(0)
    model.fit(x_train,y_train-x_train.q_nanping_lag0)
    return (x_pred.q_nanping_lag0.to_numpy()+model.predict(x_pred)).clip(0)


def score(y,pred):
    e = np.asarray(pred)-np.asarray(y)
    peak = y >= np.quantile(y,.95)
    return dict(n=len(y),rmse=float(np.mean(e**2)**.5),mae=float(np.abs(e).mean()),
                peak_rmse=float(np.mean((e[peak])**2)**.5))


def validate(args):
    xs,y = build_tables(args.data,"2021-12-31")
    oof_y = y.loc["2018":"2021"]
    all_scores,predictions,yearly = [],{},[]
    for name,x in xs.items():
        for alpha in (.03,.1,.3,1.,3.,10.,30.,100.,300.):
            preds = []
            for year in range(2018,2022):
                train = y.index.year < year
                valid = y.index.year == year
                p = fit_predict(name,alpha,x[train],y[train],x[valid])
                preds.append(p)
                yearly.append(dict(family=name,alpha=alpha,year=year,**score(y[valid],p)))
            pred = np.concatenate(preds)
            all_scores.append(dict(family=name,alpha=alpha,**score(oof_y,pred)))
            predictions[name,alpha] = pred
    scores = pd.DataFrame(all_scores).sort_values("rmse")
    winners = scores.drop_duplicates("family")
    best = winners.iloc[0]
    lock = {"family":str(best.family),"alpha":float(best.alpha),"oof_rmse":float(best.rmse),
            "last_selection_label":"2021-12-31","candidate_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "per_family":dict(zip(winners.family,winners.alpha,strict=True))}
    (args.out/"lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    frame = pd.DataFrame({"y":oof_y,"prediction":predictions[best.family,best.alpha]})
    for row in winners.itertuples():
        frame[row.family] = predictions[row.family,row.alpha]
    frame.rename_axis("target_date").to_csv(args.out/"oof.csv")
    scores.to_csv(args.out/"selection.csv",index=False)
    pd.DataFrame(yearly).to_csv(args.out/"folds.csv",index=False)
    print(winners.to_string(index=False))


def evaluate(args):
    lock = json.loads((args.out/"lock.json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=lock["candidate_code_sha256"]:
        raise ValueError("Source differs from the validation lock")
    xs,y = build_tables(args.data,"2023-12-31")
    train, test = y.index.year<2022,y.index.year>=2022
    out = pd.DataFrame({"y":y[test],"persistence":xs["group"].q_nanping_lag0[test]})
    for name,alpha in lock["per_family"].items():
        out[name] = fit_predict(name,alpha,xs[name][train],y[train],xs[name][test])
    for name in ("group","v2"):
        out[name+"_old_fixed"] = fit_predict(name,.1,xs[name][train],y[train],xs[name][test])
    out["prediction"] = out[lock["family"]]
    out.rename_axis("target_date").to_csv(args.out/"predictions.csv")
    scores = pd.DataFrame([dict(model=c,**score(out.y,out[c])) for c in out if c!="y"])
    scores.to_csv(args.out/"metrics.csv",index=False)
    print(scores.to_string(index=False))


def locked_predict(data, out, start, stop, delay_days=0):
    """Fit once before the first target, issuing predictions with past-only features."""
    out = Path(out)
    lock = json.loads((out/"lock.json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=lock["candidate_code_sha256"]:
        raise ValueError("Source differs from the validation lock")
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if start <= pd.Timestamp(lock["last_selection_label"]) or stop < start:
        raise ValueError("Select target dates after the recorded validation window")
    if not isinstance(delay_days,int) or delay_days < 0:
        raise ValueError("delay_days must be a nonnegative integer")
    xs,y = build_tables(data,stop,require_labels=False)
    dates = pd.date_range(start,stop,freq="D")
    x = xs[lock["family"]].copy()
    x.index = x.index + delay_days*DAY
    y = y.reindex(x.index)
    missing = dates.difference(x.index)
    if not missing.empty:
        raise ValueError(f"Missing inputs through the previous day for {missing[0].date()}")
    train = (x.index < start-delay_days*DAY) & y.notna()
    pred = fit_predict(lock["family"],lock["alpha"],x[train],y[train],x.loc[dates])
    return pd.Series(pred,index=dates.rename("target_date"),name="prediction")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["validate","evaluate","predict"])
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--out",type=Path,default=ROOT/"routing_results")
    p.add_argument("--target-date",help="Single target day for predict, including the day after the last label")
    p.add_argument("--delay-days",type=int,default=0,help="Additional input and training-label delay in days")
    args = p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(limits=2):
        if args.stage=="predict":
            if not args.target_date:
                p.error("predict needs --target-date YYYY-MM-DD")
            print(locked_predict(args.data,args.out,args.target_date,args.target_date,args.delay_days).to_csv(),end="")
        else:
            (validate if args.stage=="validate" else evaluate)(args)
