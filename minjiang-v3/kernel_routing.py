"""Kernel ridge with flow-dependent response coefficients and a persistence offset."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scale_routing import DATA, ROOT, DAY, design


def matrices(f, fit, ev):
    condition = f[[c for c in f if c.endswith(("_rising","_share","_wetness")) or c in ("flow_size","antecedent")]]
    driver = f[[c for c in f if "_by_" not in c and not c.endswith("_sq")]]
    def standardize(x):
        normal = StandardScaler().fit(x[fit])
        return normal.transform(x[fit]), normal.transform(x[ev])
    st,sv = standardize(condition)
    dt,dv = standardize(driver)
    # Smoothly change the linear transfer coefficients according to the current
    # hydrograph. Separate an unconditional linear term for extrapolation.
    return ((1+dt@dt.T/dt.shape[1], cdist(st,st,"sqeuclidean")/st.shape[1]),
            (1+dv@dt.T/dt.shape[1], cdist(sv,st,"sqeuclidean")/st.shape[1]))


def predict_fold(f,s,q,y,fit,ev,options):
    tr,te = matrices(f,fit,ev)
    w = (s[fit]/s[fit].mean()).to_numpy()**2
    response = ((y[fit]-q[fit])/s[fit]).to_numpy()
    for bandwidth in sorted({h for h,p in options}):
        k = tr[0]*(.25 + np.exp(-tr[1]/bandwidth))
        cross = te[0]*(.25 + np.exp(-te[1]/bandwidth))
        diag = np.diag_indices_from(k)
        for h, penalty in options:
            if h == bandwidth:
                regularized = k.copy()
                regularized[diag] += penalty / w
                coef = cho_solve(cho_factor(regularized,check_finite=False),response,check_finite=False)
                yield (h,penalty),(q[ev]+s[ev]*(cross@coef)).clip(lower=0)


def validate(data,out):
    f,s,q = design(data,"2021-12-31")
    y = q.shift(-1)
    d = f.index + DAY
    ok = f.notna().all(axis=1) & y.notna() & (d >= "2014-02-01")
    options = [(h,p) for h in (1.,4.,16.) for p in (.03,.3,3.)]
    predictions = {pair:[] for pair in options}
    for year in (2018,2019,2020,2021):
        for pair,pred in predict_fold(f,s,q,y,ok&(d.year<year),ok&(d.year==year),options):
            predictions[pair].append(pred)
    rows = []
    merged = {pair:pd.concat(pieces) for pair,pieces in predictions.items()}
    for (h,p),pred in merged.items():
        err = pred - y.loc[pred.index]
        rows.append(dict(bandwidth=h,penalty=p,rmse=np.sqrt(np.mean(err**2)),mae=err.abs().mean()))
    result = pd.DataFrame(rows).sort_values("rmse")
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"kernel_validation.csv",index=False)
    best = result.iloc[0]
    choice = dict(bandwidth=float(best.bandwidth),penalty=float(best.penalty))
    (out/"kernel_choice.json").write_text(json.dumps(choice),encoding="utf-8")
    pred = merged[(best.bandwidth,best.penalty)]
    pd.DataFrame({"target_date":pred.index+DAY,"y":y[pred.index],"prediction":pred}).set_index("target_date").to_csv(out/"kernel_oof.csv")
    print(result.to_string(index=False))


def locked_predict(data,out,start,stop,delay_days=0):
    choice = json.loads((out/"kernel_choice.json").read_text(encoding="utf-8"))
    horizon = 1 + delay_days
    f,s,q = design(data,pd.Timestamp(stop)-horizon*DAY)
    d = f.index + horizon*DAY
    y = q.shift(-horizon)
    ok = f.notna().all(axis=1)
    fit = ok & y.notna() & (d >= "2014-02-01") & (d <= pd.Timestamp(start)-horizon*DAY)
    ev = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(stop))
    if not ok[ev].all() or not ev.any():
        raise ValueError("Incomplete evaluation predictors")
    _,pred = next(predict_fold(f,s,q,y,fit,ev,[(choice["bandwidth"],choice["penalty"])]))
    pred.index = pred.index + horizon*DAY
    return pred.rename("prediction").rename_axis("target_date")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=ROOT/"experiments")
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        validate(args.data,args.out)
