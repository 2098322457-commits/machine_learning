"""Flow transformation and a river-transfer regression with autoregressive errors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"minjiang-v3"))
from routing import DEFAULT_DATA, build_tables


def forecast_one(table, labels, start, stop, power, penalty, order):
    """Daily issue at end of previous day. The regression is fitted before start."""
    fit=table.index < pd.Timestamp(start)
    ev=(table.index >= pd.Timestamp(start)) & (table.index <= pd.Timestamp(stop))
    if labels[fit].isna().any() or fit.sum() < 365:
        raise ValueError("Insufficient historical targets")
    level=table.q_nanping_lag0.to_numpy()+50
    offset=level**power
    response=(labels.to_numpy()+50)**power-offset
    importance=level[fit]**(1-power)
    importance/=importance.mean()
    x=table.to_numpy()*level[:,None]**(power-1)
    normal=StandardScaler().fit(x[fit],sample_weight=importance)
    z=normal.transform(x)
    model=Ridge(alpha=penalty,solver="cholesky")
    model.fit(z[fit],response[fit],sample_weight=importance)
    a=np.zeros(order)
    intercept_adjustment=0.
    if order:
        xt,yt=z[fit],response[fit]
        for _ in range(3):
            resid=yt-model.predict(xt)-intercept_adjustment
            lags=np.column_stack([resid[order-k:-k] for k in range(1,order+1)])
            a=np.linalg.lstsq(lags,resid[order:],rcond=None)[0]
            a*=min(1.,.85/max(np.abs(a).sum(),1.e-9))
            white_x=xt[order:].copy()
            white_y=yt[order:].copy()
            for k,coef in enumerate(a,1):
                white_x-=coef*xt[order-k:-k]
                white_y-=coef*yt[order-k:-k]
            model.fit(white_x,white_y,sample_weight=importance[order:])
            intercept_adjustment=model.intercept_*a.sum()/(1-a.sum())
    delta=model.predict(z)+intercept_adjustment
    residual=pd.Series(response-delta,index=table.index)
    for k,coef in enumerate(a,1):
        delta+=coef*residual.shift(k).fillna(0).to_numpy()
    pred=np.maximum(np.maximum(offset+delta,0)**(1/power)-50,0)
    return pd.Series(pred[ev],index=table.index[ev],name="prediction")


def validate(data,out):
    out=Path(out)
    out.mkdir(exist_ok=True,parents=True)
    families,y=build_tables(data,"2020-12-31")
    rows=[]
    all_pred=[]
    for family in ("storm","channel_storm"):
        x=families[family]
        for power in (.5,.75,1.):
            for penalty in (10,100,500):
                for order in (0,2):
                    kwargs=dict(power=power,penalty=penalty,order=order)
                    val=pd.concat([forecast_one(x,y,f"{year}-01-01",f"{year}-12-31",**kwargs)
                                   for year in (2018,2019,2020)])
                    err=val-y.loc[val.index]
                    rows.append(dict(family=family,**kwargs,rmse=float(np.sqrt(np.mean(err**2))),
                                     mae=float(abs(err).mean())))
                    all_pred.append(val)
    ranked=pd.DataFrame(rows).sort_values("rmse")
    ranked.to_csv(out/"ar_validation.csv",index=False)
    best=rows[int(ranked.index[0])]
    lock={k:v for k,v in best.items() if k not in ("rmse","mae")}
    (out/"ar_lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    pred=all_pred[int(ranked.index[0])]
    pd.DataFrame({"y":y.loc[pred.index],"prediction":pred}).rename_axis("target_date").to_csv(out/"ar_oof.csv")
    print(ranked.head(5).to_string(index=False))


def locked_predict(data,out,start,stop):
    config=json.loads((Path(out)/"ar_lock.json").read_text())
    tables,y=build_tables(data,stop,require_labels=False)
    return forecast_one(tables[config.pop("family")],y,start,stop,**config)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--out",type=Path,default=ROOT/"ar")
    args=parser.parse_args()
    with threadpool_limits(limits=3):
        validate(args.data,args.out)
