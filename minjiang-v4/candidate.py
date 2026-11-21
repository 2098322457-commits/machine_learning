"""Daily Nanping flow from a global routing equation and local hydrograph analogues."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "minjiang-original" / "data"
DAY = pd.Timedelta(days=1)
SITES = ["nanping","jianou","sanming","shaowu","shuikou_dam"]
BASINS = {"jianxi":["pucheng","wuyishan","jianou","jianyang"],
          "futun":["shaowu","guangze","taining","jiangle"],
          "shaxi":["sanming","yongan","youxi"],"main":["nanping"]}


def inputs(data, end):
    """All returned rows use readings through their source day, inclusive."""
    end = pd.Timestamp(end).normalize()
    flow = pd.read_csv(Path(data)/"glofas_daily_discharge.csv",parse_dates=["date"])
    flow = flow.loc[flow.date<=end].pivot(index="date",columns="station",values="river_discharge_m3s")
    flow = flow.sort_index().asfreq("D")
    raw = pd.read_csv(Path(data)/"openmeteo_hourly_precipitation.csv",parse_dates=["time"],index_col="time")
    rain = raw.loc[raw.index<end+DAY].sort_index().asfreq("h")
    q = flow.nanping
    scale = 75+.55*q+.45*q.rolling(4).mean()
    state, driver, gating = {}, {}, {}
    for site in SITES:
        s = flow[site]
        recent = s.rolling(5).mean()
        state[site+"_rise"] = np.log((s+20)/(recent+20))
        state[site+"_rate"] = np.log((s+20)/(s.shift(1)+20))
        state[site+"_mix"] = np.log((s+50)/(q+100))
        driver[site+"_level"] = s/scale
        for lag in (0,1,2,3):
            driver[f"{site}_delta{lag}"] = s.diff().shift(lag)/scale
        gating[site+"_rise"] = s/(s+recent+15)
    state["stage"] = np.log(scale)
    state["antecedent"] = np.log((q.rolling(14).mean()+100)/(q+100))
    gating["stage"] = q/(q+600)
    local_rain = {}
    for basin,sites in BASINS.items():
        h = rain[sites].mean(axis=1,skipna=False)
        daily = h.resample("D").sum(min_count=24)
        wet = daily.rolling(10).mean().shift(1)
        state[basin+"_rain"] = np.log1p(daily)
        state[basin+"_wet"] = np.log1p(wet)
        for lag in (0,1,2):
            driver[f"{basin}_rain{lag}"] = daily.shift(lag)/scale
        for start in (0,6,12,18):
            segment = h[(h.index.hour//6)*6==start].resample("D").sum(min_count=6)
            driver[f"{basin}_hours{start}"] = segment/scale
        local_rain[basin] = daily*wet/(5+wet)/scale
    state, driver = pd.DataFrame(state),pd.DataFrame(driver)
    # Cubic Bernstein bases allow the global routing slopes to vary smoothly
    # with stage and with the rising/falling phase of each river hydrograph.
    interactions = {}
    core = [c for c in driver if c.endswith(("_level","_delta0","_delta1","_rain0"))]
    for key,u in gating.items():
        for degree,b in enumerate([3*u*(1-u)**2,3*u**2*(1-u),u**3],start=1):
            for col in core:
                interactions[f"{key}_basis{degree}_{col}"] = b*driver[col]
    full = pd.concat([driver,state,pd.DataFrame(local_rain).add_prefix("runoff_"),
                      pd.DataFrame(interactions)],axis=1)
    local = driver[[c for c in driver if c.endswith(("_level","_delta0","_delta1","_rain0","_rain1"))]]
    return full, local, state, scale, q


def predict_options(tables, train, ev, y, options):
    full,local,state,scale,q = tables
    st = StandardScaler().fit(state[train])
    sx = StandardScaler().fit(local[train])
    x_local, x_query = sx.transform(local[train]),sx.transform(local[ev])
    # Levels and recession shape stay informative across different storm sizes.
    state_weights = np.array([0.45 if c.endswith(("_rain","_wet")) else 1 for c in state])
    dist = cdist(st.transform(state[ev])*state_weights,st.transform(state[train])*state_weights,"sqeuclidean")
    kmax = min(max(k for _,k,_ in options),len(x_local))
    neighbor = np.argsort(dist,axis=1)[:,:kmax]
    response = ((y[train]-q[train])/scale[train]).to_numpy()
    weights = (scale[train]/scale[train].mean()).to_numpy()**1.5
    weights /= weights.mean()
    normal = StandardScaler().fit(full[train],sample_weight=weights)
    xf,xe = normal.transform(full[train]),normal.transform(full[ev])
    for alpha in sorted({a for a,_,_ in options}):
        global_fit = Ridge(alpha=alpha,solver="cholesky").fit(xf,response,sample_weight=weights)
        center = global_fit.predict(xe)
        residual = response-global_fit.predict(xf)
        for _,k,penalty in [c for c in options if c[0]==alpha]:
            correction = np.zeros(len(center))
            if k:
                for i,index in enumerate(neighbor[:,:k]):
                    local_dist = dist[i,index]
                    w = np.exp(-2*local_dist/(local_dist[-1]+1e-10))
                    w /= w.sum()
                    x = x_local[index]
                    z = residual[index]
                    mx, mz = w@x, w@z
                    xc = x-mx
                    weighted = xc.T*w
                    matrix = weighted@xc
                    matrix.flat[::len(mx)+1] += penalty
                    slope = solve(matrix,weighted@(z-mz),assume_a="pos",check_finite=False)
                    correction[i] = mz+(x_query[i]-mx)@slope
            pred = (q[ev]+scale[ev]*(center+correction)).clip(lower=0)
            yield (alpha,k,penalty),pred


def validate(data, out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    tables = inputs(data,"2020-12-31")
    full,_,_,_,q = tables
    y = q.shift(-1)
    day = q.index+DAY
    good = full.notna().all(axis=1)&y.notna()&(day>=pd.Timestamp("2014-02-01"))
    options = [(a,0,0) for a in (10,100,500)]
    options.extend((a,k,p) for a in (10,100,500) for k in (60,180,450) for p in (.03,.3))
    by_opt = {c:[] for c in options}
    folds = []
    for year in (2018,2019,2020):
        train = good & (day.year<year)
        ev = good & (day.year==year)
        for config,pred in predict_options(tables,train,ev,y,options):
            by_opt[config].append(pred)
            folds.append(dict(alpha=config[0],neighbors=config[1],penalty=config[2],year=year,
                              n=len(pred),rmse=np.mean((pred-y.loc[pred.index])**2)**.5))
    scores = []
    for (alpha,k,p),pieces in by_opt.items():
        pred = pd.concat(pieces)
        error = pred-y.loc[pred.index]
        scores.append(dict(alpha=alpha,neighbors=k,penalty=p,n=len(error),rmse=np.sqrt(np.mean(error**2)),
                           mae=error.abs().mean()))
    table = pd.DataFrame(scores).sort_values("rmse")
    best = table.iloc[0]
    cfg = (float(best.alpha),int(best.neighbors),float(best.penalty))
    pred = pd.concat(by_opt[cfg])
    table.to_csv(out/"candidate_validation.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"candidate_folds.csv",index=False)
    pd.DataFrame({"target_date":pred.index+DAY,"y":y.loc[pred.index].to_numpy(),
                  "prediction":pred.to_numpy()}).to_csv(out/"candidate_oof.csv",index=False)
    lock = dict(alpha=cfg[0],neighbors=cfg[1],penalty=cfg[2],oof_rmse=float(best.rmse),
                last_selection_label="2020-12-31",sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"candidate_choice.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(table.to_string(index=False))


def locked_predict(data,out,start,stop,delay_days=0):
    """Fit once before start, returning predictions indexed by target dates."""
    start,stop = pd.Timestamp(start),pd.Timestamp(stop)
    if stop<start or not isinstance(delay_days,int) or delay_days<0:
        raise ValueError("Invalid target dates or source delay")
    h = 1+delay_days
    choice = json.loads((Path(out)/"candidate_choice.json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=choice["sha256"]:
        raise ValueError("Model source differs from frozen validation")
    if start-h*DAY < pd.Timestamp(choice["last_selection_label"]):
        raise ValueError("Validation labels unavailable at first issue")
    tables = inputs(data,stop-h*DAY)
    full,_,_,_,q = tables
    day = q.index + h*DAY
    y = q.shift(-h)
    good = full.notna().all(axis=1)
    train = good & y.notna() & (day>=pd.Timestamp("2014-02-01")) & (day<=start-h*DAY)
    ev = (day>=start)&(day<=stop)
    if not good[ev].all() or not day[ev].equals(pd.date_range(start,stop,freq="D")):
        raise ValueError("Incomplete forecasting inputs")
    cfg = [(choice["alpha"],choice["neighbors"],choice["penalty"])]
    _,pred = next(predict_options(tables,train,ev,y,cfg))
    pred.index += h*DAY
    return pred.rename_axis("target_date").rename("prediction")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["validate","predict"])
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=ROOT/"candidate_results")
    parser.add_argument("--start")
    parser.add_argument("--stop")
    parser.add_argument("--delay-days",type=int,default=0)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage=="validate":
            validate(args.data,args.out)
        else:
            if not args.start or not args.stop:
                parser.error("predict needs --start and --stop")
            print(locked_predict(args.data,args.out,args.start,args.stop,args.delay_days).to_csv(),end="")
