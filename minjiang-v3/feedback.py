"""Daily, strictly lagged error calibration. Select only on pre-2022 out-of-fold errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from threadpoolctl import threadpool_limits
from run import ROOT, forecast, metrics
from routing import DEFAULT_DATA, build_tables, fit_predict


def history(out):
    policy = json.loads((out/"frozen.json").read_text())["weights"]
    raw = pd.read_csv(out/"routing_results/oof.csv",index_col="target_date",parse_dates=True)
    pred = policy["channel"] * raw.prediction
    for name in ("scale","kernel"):
        extra = pd.read_csv(out/"experiments"/f"{name}_oof.csv",index_col="target_date",parse_dates=True)
        pred = pred + policy[name]*extra.prediction
    return pd.DataFrame(dict(target=raw.y,base=pred))


def covariates(frame):
    prior = frame.target.shift(1)
    error = (frame.target-frame.base).shift(1)
    scale = prior.abs()+100
    rise = ((frame.base-prior)/scale).clip(-1,1)
    x = pd.DataFrame({"e1":error,"e2":error.shift(1),"e3":error.shift(2),
                      "avg3":error.rolling(3).mean(),"avg7":error.rolling(7).mean(),
                      "e1_rising":error*rise,"e2_rising":error.shift(1)*rise,
                      "signed_e1":error.abs()*rise,"e1_stage":error*np.log1p(prior/500)})
    return x.fillna(0)


def adjusted(frame,fit,ev,alpha):
    if alpha=="none":
        return frame.base.loc[ev]
    x = covariates(frame)
    z = StandardScaler().fit(x.loc[fit])
    reg = Ridge(alpha=float(alpha)).fit(z.transform(x.loc[fit]),(frame.target-frame.base).loc[fit])
    result = frame.base.loc[ev] + reg.predict(z.transform(x.loc[ev]))
    return result.clip(lower=0)


def choose_alpha(frame):
    rows = []
    for alpha in ("none","30","300","1500"):
        segments=[]
        for year in (2019,2020,2021):
            segments.append(adjusted(frame,frame.index.year<year,frame.index.year==year,alpha))
        pred = pd.concat(segments)
        rows.append(dict(alpha=alpha,**metrics(frame.target.loc[pred.index],pred)))
    ranked = pd.DataFrame(rows).sort_values("rmse")
    return ranked


def validate(out,data):
    xs,y = build_tables(data,"2021-12-31")
    histories = {"v3":history(out)}
    for family in ("group","v2"):
        forecasts=[]
        for year in (2018,2019,2020,2021):
            tr,va = y.index.year<year,y.index.year==year
            forecasts.append(pd.Series(fit_predict(family,.1,xs[family][tr],y[tr],xs[family][va]),
                                       index=y[va].index))
        series=pd.concat(forecasts)
        histories[family]=pd.DataFrame({"target":y.loc[series.index],"base":series})
    policies,logs={},[]
    for name,frame in histories.items():
        ranked=choose_alpha(frame)
        policies[name]=str(ranked.iloc[0].alpha)
        logs.append(ranked.assign(model=name))
        frame.rename_axis("target_date").to_csv(out/f"feedback_history_{name}.csv")
    pd.concat(logs).to_csv(out/"feedback_validation.csv",index=False)
    (out/"feedback_choice.json").write_text(json.dumps(policies),encoding="utf-8")
    print("Validation choices:",policies)


def validate_archive(history_frame,future_frame):
    idx = history_frame.index.append(future_frame.index)
    if not idx.equals(pd.date_range(idx[0],idx[-1],freq="D")):
        raise ValueError("Target dates must form a consecutive archive")


def backtest(out):
    previous = history(out)
    scores = pd.read_csv(out/"test_predictions.csv",index_col="target_date",parse_dates=True)
    recent = scores[["target","v3"]].rename(columns={"v3":"base"})
    validate_archive(previous,recent)
    combined = pd.concat([previous,recent])
    policy = json.loads((out/"feedback_choice.json").read_text())
    alpha = policy["v3"]
    fit,ev = combined.index.year<2022,combined.index.year>=2022
    pred = adjusted(combined,fit,ev,alpha)
    # Changing a target may only affect forecasts on strictly later dates.
    changed = combined.copy()
    change_day = pd.Timestamp("2022-06-01")
    changed.loc[change_day:,"target"] += 200
    other = adjusted(changed,fit,ev,alpha)
    np.testing.assert_allclose(other.loc[:change_day],pred.loc[:change_day])
    models = {"v3":recent.base,"v3_feedback":pred}
    for name in ("group","v2"):
        hist=pd.read_csv(out/f"feedback_history_{name}.csv",index_col="target_date",parse_dates=True)
        obs=recent.assign(base=scores[name+"_old_fixed"])
        full=pd.concat([hist,obs])
        models[name]=obs.base
        models[name+"_feedback"]=adjusted(full,fit,ev,policy[name])
    pd.DataFrame(models).to_csv(out/"feedback_predictions.csv")
    rows = [dict(model=name,**metrics(recent.target,p)) for name,p in models.items()]
    pd.DataFrame(rows).to_csv(out/"feedback_test.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def predict_day(out,data,day):
    # Rebuild the same fixed-model issue history as the benchmark. Only already
    # completed targets may enter error feedback; the requested target is masked.
    current=pd.Timestamp(day)
    base=forecast(data,out,"2022-01-01",str(current.date()),0).v3
    raw=pd.read_csv(data/"glofas_daily_discharge.csv",parse_dates=["date"])
    actual=raw.loc[(raw.station=="nanping") & (raw.date<current)].set_index("date").river_discharge_m3s
    recent=pd.DataFrame({"base":base,"target":actual.reindex(base.index)})
    if recent.target.iloc[:-1].isna().any():
        raise ValueError("Missing prior daily flow required for feedback")
    full=pd.concat([history(out),recent])
    alpha=json.loads((out/"feedback_choice.json").read_text())["v3"]
    pred=adjusted(full,full.index.year<2022,full.index==current,alpha)
    return pred.rename("nanping_m3s")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["validate","evaluate","predict"])
    parser.add_argument("--out",type=Path,default=ROOT)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--date")
    args=parser.parse_args()
    with threadpool_limits(limits=3):
        if args.stage=="validate":
            validate(args.out,args.data)
        elif args.stage=="evaluate":
            backtest(args.out)
        else:
            if not args.date:
                parser.error("--date required")
            print(predict_day(args.out,args.data,args.date).to_csv(),end="")
