"""Nanping next-day flow with stage-dependent travel times and rainfall pulse routing."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent/"minjiang-original"/"data"
DAY = pd.Timedelta(days=1)
SITES = ("nanping","jianou","sanming","shaowu","shuikou_dam")
BASINS = {"jianxi":["pucheng","wuyishan","jianou","jianyang"],
          "futun":["shaowu","guangze","taining","jiangle"],
          "shaxi":["sanming","yongan","youxi"],"main":["nanping"]}


def read_flow(data,end):
    raw = pd.read_csv(Path(data)/"glofas_daily_discharge.csv",parse_dates=["date"])
    raw = raw.loc[raw.date<=end]
    return raw.pivot(index="date",columns="station",values="river_discharge_m3s").sort_index().asfreq("D")


def read_hours(data,var,end):
    raw = pd.read_csv(Path(data)/f"openmeteo_hourly_{var}.csv",index_col="time",parse_dates=["time"])
    return raw.loc[raw.index<end+DAY].sort_index().asfreq("h")


def future_rain_response(hourly, tau):
    """Mean response in the NEXT 24 h to ONLY the rain already observed.

    For two cascaded first-order reservoirs the zero-input response at step j
    equals a**j * (second + j * (1-a) * first), both states at the issue hour.
    """
    a = np.exp(-1/tau)
    first = hourly.ewm(alpha=1-a,adjust=False).mean()
    second = first.ewm(alpha=1-a,adjust=False).mean()
    lead = np.arange(1,25)
    response = (a**lead).mean()*second + ((a**lead)*lead*(1-a)).mean()*first
    return response.resample("D").last()


def pulse_inputs(data,end,scale):
    rain = read_hours(data,"precipitation",end)
    result,state = {},{}
    for basin,stations in BASINS.items():
        h = rain[stations].mean(axis=1,skipna=False)
        daily = h.resample("D").sum(min_count=24)
        recent = daily.ewm(halflife=10,adjust=False).mean()
        wet = recent/(recent+6)
        state[basin+"_wet"] = wet
        for tau in (8,20,48,100):
            pulse = 24*future_rain_response(h,tau)/scale
            result[f"{basin}_pulse{tau}"] = pulse
            result[f"{basin}_wetpulse{tau}"] = pulse*wet
        for lag in (0,1,2):
            result[f"{basin}_rain{lag}"] = daily.shift(lag)/scale
        last = h[h.index.hour>=18].resample("D").sum(min_count=6)/scale
        result[basin+"_last6"] = last
        result[basin+"_wet6"] = last*wet
    # These are historical within-day weather tendencies, never realised
    # weather on the target day. They can indicate continuation of a storm.
    for var in ("relative_humidity_2m","surface_pressure"):
        s = read_hours(data,var,end).mean(axis=1)
        early = s[s.index.hour<6].resample("D").mean()
        late = s[s.index.hour>=18].resample("D").mean()
        result[var+"_change"] = late-early
        result[var+"_yesterday"] = late-late.shift(1)
    return pd.DataFrame(result),pd.DataFrame(state)


def arrival(s,travel):
    """Fractional-day arrival at tomorrow's average observation time."""
    # τ days of propagation puts the source at today + 1 - τ. For τ>=1,
    # bracket only historical daily averages and linearly interpolate them.
    back = np.maximum(travel.to_numpy()-1,0)
    whole = np.nan_to_num(np.floor(back),nan=0).astype(int)
    frac = back-whole
    past = np.stack([s.shift(lag).to_numpy() for lag in range(6)],axis=1)
    row = np.arange(len(s))
    interp = (1-frac)*past[row,whole]+frac*past[row,whole+1]
    # For a sub-day travel time the source would be in the future. Project a
    # fraction of the observed slope, bounded relative to the current level.
    slope = s.diff().clip(lower=-.6*(s+25),upper=.95*(s+25))
    ahead = np.maximum(1-travel.to_numpy(),0)
    extrap = s + (1-np.exp(-1.5*ahead))/1.5*slope
    return pd.Series(np.where(back>0,interp,extrap),index=s.index)


def design(q, pulses, wet, gamma):
    n = q.nanping
    scale = 90+.65*n+.35*n.rolling(5).mean()
    state, inputs, selected = {},{},[]
    state["stage"] = n/(n+750)
    state["ratio5"] = np.log((n+40)/(n.rolling(5).mean()+40))
    for site in SITES:
        s = q[site]
        slow = s.rolling(24).mean()
        intensity = np.sqrt(((s+40)/(slow+40))*(scale/600)).clip(.12,12)
        state[site+"_rise"] = s.diff()/(s+s.shift(1)+40)
        state[site+"_mix"] = s/(s+n+80)
        inputs[site+"_level"] = s/scale
        for lag in (0,1,2,3):
            name = f"{site}_change{lag}"
            inputs[name] = s.diff().shift(lag)/scale
            if lag<=1:
                selected.append(name)
        # Reference times span fast/slow paths and are basis functions,
        # not surveyed river travel times. gamma controls wave-speed response.
        for tau in (.5,1,1.75,2.75,4):
            travel = (tau*intensity**(-gamma)).clip(.25,4.5)
            name = f"{site}_arrive{tau:g}"
            inputs[name] = (arrival(s,travel)-s)/scale
            selected.append(name)
        selected.append(site+"_level")
    drivers = pd.concat([pd.DataFrame(inputs),pulses],axis=1)
    state = pd.DataFrame(state)
    selected += [col for col in pulses if col.endswith(("pulse20","pulse48")) and "wet" not in col]
    # Stage, rising/receding phase and tributary mix modulate routing slopes.
    # The ridge fit shares the information from quiet days and flood events.
    cross = {f"{name}_x_{condition}":drivers[name]*state[condition]
             for name in selected for condition in state}
    for name in selected:
        cross[name+"_stage2"] = drivers[name]*state.stage**2
        cross[name+"_rise2"] = drivers[name]*state.nanping_rise.abs()
    features = pd.concat([drivers,state,wet,pd.DataFrame(cross)],axis=1)
    return features,scale


def fit_forecast(x,scale,q,y,tr,ev,alpha,power):
    current = q.nanping
    # power=1 is the next increment on the discharge scale; power=0.5
    # emphasizes differences on a storage-like square-root scale.
    offset = 60.
    response = ((y+offset)**power-(current+offset)**power) / scale**power
    weight = scale[tr].to_numpy()**(1.4*power)
    weight /= weight.mean()
    standard = StandardScaler().fit(x[tr],sample_weight=weight)
    model = Ridge(alpha=alpha,solver="cholesky").fit(standard.transform(x[tr]),response[tr],sample_weight=weight)
    change = model.predict(standard.transform(x[ev]))
    back = (current[ev]+offset)**power+change*scale[ev]**power
    return (back.clip(lower=0)**(1/power)-offset).clip(lower=0)


def validate(data,artifacts):
    out = Path(artifacts)
    out.mkdir(parents=True,exist_ok=True)
    cutoff = pd.Timestamp("2020-12-31")
    q = read_flow(data,cutoff)
    scale = 90+.65*q.nanping+.35*q.nanping.rolling(5).mean()
    pulses,wet = pulse_inputs(data,cutoff,scale)
    y = q.nanping.shift(-1)
    days = q.index+DAY
    scores,folds,preds = [],[],{}
    for gamma in (0.,.3,.65):
        x,scale = design(q,pulses,wet,gamma)
        available = x.notna().all(axis=1)&y.notna()&(days>=pd.Timestamp("2014-02-01"))
        for alpha in (10.,60.,250.,900.):
            for power in (.5,1.):
                key = (gamma,alpha,power)
                yearly = []
                for year in (2018,2019,2020):
                    tr = available & (days.year<year)
                    ev = available & (days.year==year)
                    p = fit_forecast(x,scale,q,y,tr,ev,alpha,power)
                    err = p-y.loc[p.index]
                    folds.append(dict(gamma=gamma,alpha=alpha,power=power,year=year,n=len(p),
                                      rmse=float(np.mean(err**2)**.5)))
                    yearly.append(p)
                merged = pd.concat(yearly)
                err = merged-y.loc[merged.index]
                preds[key] = merged
                scores.append(dict(gamma=gamma,alpha=alpha,power=power,n=len(err),
                                   rmse=float(np.mean(err**2)**.5),mae=float(err.abs().mean())))
    table = pd.DataFrame(scores).sort_values("rmse")
    best = table.iloc[0]
    cfg = (float(best.gamma),float(best.alpha),float(best.power))
    p = preds[cfg]
    oof = pd.DataFrame({"target_date":p.index+DAY,"target":y.loc[p.index].to_numpy(),
                        "prediction":p.to_numpy()})
    if not pd.DatetimeIndex(oof.target_date).equals(pd.date_range("2018-01-01",cutoff,freq="D")):
        raise ValueError("Validation target coverage differs from the expected grid")
    choice = dict(gamma=cfg[0],alpha=cfg[1],power=cfg[2],last_selection_target="2020-12-31",
                  oof_rmse=float(best.rmse),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    oof.to_csv(out/"oof.csv",index=False)
    table.to_csv(out/"selection.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"folds.csv",index=False)
    (out/"choice.json").write_text(json.dumps(choice,indent=2),encoding="utf-8")
    print(table.head(10).to_string(index=False))


def predict(data,artifacts,start,stop):
    """Fit a single frozen-parameter model; daily inputs stop at each target's previous day."""
    start,stop = pd.Timestamp(start),pd.Timestamp(stop)
    choice = json.loads((Path(artifacts)/"choice.json").read_text(encoding="utf-8"))
    if start<=pd.Timestamp(choice["last_selection_target"]) or stop<start:
        raise ValueError("Target range precedes freeze or has reversed endpoints")
    if choice["sha256"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Source differs from the pretest freeze")
    end = stop-DAY
    q = read_flow(data,end)
    scale = 90+.65*q.nanping+.35*q.nanping.rolling(5).mean()
    pulses,wet = pulse_inputs(data,end,scale)
    x,scale = design(q,pulses,wet,choice["gamma"])
    day = q.index+DAY
    y = q.nanping.shift(-1)
    good = x.notna().all(axis=1)
    fit = good & y.notna() & (day>=pd.Timestamp("2014-02-01")) & (day<start)
    ev = (day>=start)&(day<=stop)
    if not good[ev].all() or not day[ev].equals(pd.date_range(start,stop,freq="D")):
        raise ValueError("Incomplete history for target dates")
    p = fit_forecast(x,scale,q,y,fit,ev,choice["alpha"],choice["power"])
    p.index += DAY
    return p.rename_axis("target_date").rename("prediction")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["validate","predict"])
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--artifacts",type=Path,default=ROOT/"transfer")
    parser.add_argument("--start")
    parser.add_argument("--stop")
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage=="validate":
            validate(args.data,args.artifacts)
        else:
            if not args.start or not args.stop:
                parser.error("predict requires --start and --stop")
            print(predict(args.data,args.artifacts,args.start,args.stop).to_csv(),end="")
