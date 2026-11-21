"""Scale-adjusted daily flow transfer from the team's raw data only."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "minjiang-original" / "data"
DAY = pd.Timedelta(days=1)
BASINS = {
    "jianxi": ["pucheng", "wuyishan", "jianyang", "jianou"],
    "futun": ["shaowu", "guangze", "taining", "jiangle"],
    "shaxi": ["sanming", "yongan", "youxi"],
    "nanping": ["nanping"],
}


def design(data, end=None):
    """Rows are indexed by issue day. Today's daily means require a completed day."""
    flow = pd.read_csv(data / "glofas_daily_discharge.csv", parse_dates=["date"])
    q = flow.pivot(index="date", columns="station", values="river_discharge_m3s").sort_index().asfreq("D")
    rain = pd.read_csv(data/"openmeteo_hourly_precipitation.csv", index_col="time", parse_dates=["time"])
    rain = rain.sort_index().asfreq("h")
    if end is not None:
        end = pd.Timestamp(end).normalize()
        q = q.loc[:end]
        rain = rain.loc[rain.index < end + DAY]
    # Offset keeps ratios finite during very low flows. Current and antecedent
    # downstream discharge set the units for tomorrow's change in discharge.
    scale = 100 + .5 * q.nanping + .5 * q.nanping.rolling(5).mean()
    transport, state = {}, {}
    for site in q:
        s = q[site]
        r = (s + 10) / (s.rolling(5).mean() + 10)
        state[site+"_rising"] = np.log(r)
        state[site+"_share"] = np.log1p(s / scale)
        transport[site+"_level"] = s / scale
        for lag in (1,2,3,5):
            transport[f"{site}_change{lag}"] = (s.shift(lag-1) - s.shift(lag)) / scale
    state["flow_size"] = np.log(scale / 500)
    state["antecedent"] = np.log((q.nanping.rolling(14).mean()+100)/scale)
    rainfall = {}
    for name, stations in BASINS.items():
        hourly = rain[stations].mean(axis=1,skipna=False)
        daily = hourly.resample("D").sum(min_count=24)
        wet = daily.rolling(14).mean()
        state[name+"_wetness"] = np.log1p(wet)
        for lag in (0,1,2,4):
            amount = daily.shift(lag)
            rainfall[f"{name}_rain{lag}"] = amount / scale
            rainfall[f"{name}_runoff{lag}"] = amount * np.sqrt(wet+1) / scale
        for hour in (0,6,12,18):
            subtotal = hourly[(hourly.index.hour//6)*6 == hour].resample("D").sum(min_count=6)
            rainfall[f"{name}_rain_from{hour}"] = subtotal / scale
    z = pd.DataFrame(state)
    u = pd.DataFrame(transport)
    # Coefficients of the transfer equation can vary with wetness, stage and
    # whether a tributary is rising. All products use the same issue-time row.
    interaction = {f"{col}_by_{axis}": u[col]*z[axis] for col in u for axis in z}
    f = pd.concat([u,z,np.square(z).add_suffix("_sq"),pd.DataFrame(rainfall),
                   pd.DataFrame(interaction)],axis=1)
    return f, scale, q.nanping


def response_basis(features):
    """Smooth flow-regime gates for the linear rainfall/runoff response."""
    primary = features.loc[:, ~features.columns.str.contains("_by_|_sq$")]
    names = [col for col in primary if col.endswith(("_level","_change1","_change2","_rain0","_rain1"))]
    additional = {}
    for axis, centers, width in [
            ("flow_size", (-1.0,0.0,1.0,2.0), 0.6),
            ("nanping_rising", (-.3,0.1,.5), .25),
            ("jianou_rising", (0,.5), .3),
            ("sanming_rising", (0,.5), .3),
            ("shuikou_dam_rising", (0,.5), .3)]:
        for center in centers:
            gate = np.tanh((primary[axis] - center)/width)
            for name in names:
                additional[f"{axis}_{center:g}_{name}"] = gate * primary[name]
    return pd.concat([primary,pd.DataFrame(additional,index=features.index)],axis=1)


def fit_predict(features, scale, q, y, fit, evaluate, alpha, exponent):
    target = (y - q) / scale
    weights = scale[fit] ** exponent
    weights /= weights.mean()
    normal = StandardScaler().fit(features[fit],sample_weight=weights)
    model = Ridge(alpha=alpha,solver="cholesky").fit(normal.transform(features[fit]),
            target[fit],sample_weight=weights)
    delta = model.predict(normal.transform(features[evaluate]))
    return (q[evaluate] + scale[evaluate]*delta).clip(lower=0)


def validation(data, out):
    out.mkdir(exist_ok=True,parents=True)
    f,s,q = design(data, "2021-12-31")
    y = q.shift(-1)
    day = f.index + DAY
    complete = f.notna().all(axis=1) & y.notna() & (day >= "2014-02-01")
    scores, oofs = [], {}
    for kind, features in [("linear",f),("gated",response_basis(f))]:
        for alpha in (3,30,120,500,1800):
            for exponent in (0.75,1.5,2.0):
                pieces = []
                for year in (2018,2019,2020,2021):
                    fit = complete & (day.year < year)
                    valid = complete & (day.year == year)
                    pieces.append(fit_predict(features,s,q,y,fit,valid,alpha,exponent))
                pred = pd.concat(pieces)
                err = pred-y.loc[pred.index]
                name = f"{kind}_{alpha:g}_weight_{exponent:g}"
                scores.append(dict(name=name,kind=kind,alpha=alpha,exponent=exponent,n=len(err),
                                   rmse=np.sqrt(np.mean(err**2)),mae=abs(err).mean()))
                oofs[name] = pred
    table = pd.DataFrame(scores).sort_values("rmse")
    table.to_csv(out/"scale_validation.csv",index=False)
    best = table.iloc[0]
    pred = oofs[best["name"]]
    frame = pd.DataFrame({"target_date":pred.index+DAY, "y":y.loc[pred.index],
                          "prediction":pred}).set_index("target_date")
    frame.to_csv(out/"scale_oof.csv")
    (out/"scale_choice.json").write_text(json.dumps(dict(kind=str(best.kind),alpha=float(best.alpha),exponent=float(best.exponent))),encoding="utf-8")
    print(table.to_string(index=False))


def locked_predict(data, out, start, stop, delay_days=0):
    """Fit on labels preceding start. Honor stop's issue date when building inputs."""
    choice = json.loads((out/"scale_choice.json").read_text(encoding="utf-8"))
    horizon = 1 + delay_days
    f,s,q = design(data,pd.Timestamp(stop)-horizon*DAY)
    if choice.pop("kind") == "gated":
        f = response_basis(f)
    day = f.index+horizon*DAY
    # Labels available before first issue include at most the source day of
    # that issue (completed daily aggregate under our availability assumption).
    y = q.shift(-horizon)
    complete = f.notna().all(axis=1)
    fit = complete & y.notna() & (day >= "2014-02-01") & (day <= pd.Timestamp(start)-horizon*DAY)
    ev = (day >= pd.Timestamp(start)) & (day <= pd.Timestamp(stop))
    if not complete[ev].all() or not ev.any():
        raise ValueError("Missing predictors for requested target dates")
    result = fit_predict(f,s,q,y,fit,ev,**choice)
    result.index = result.index + horizon*DAY
    return result.rename_axis("target_date").rename("prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=ROOT/"experiments")
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        validation(args.data,args.out)
