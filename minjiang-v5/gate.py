"""Forward test of a pre-specified flow-dependent blending rule on V4 development OOF."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "minjiang-v4" / "results" / "validation_oof.csv"
DATA = ROOT.parent / "minjiang-original" / "data"
DAY = pd.Timedelta(days=1)
FIXED_WEIGHT = 0.5
PRIOR_DAYS = 90
MIN_GAIN = 0.01


def stage_history(data, stop):
    """Load only the Nanping observations available through the requested source date."""
    stop = pd.Timestamp(stop).strftime("%Y-%m-%d")
    rows = []
    with (Path(data)/"glofas_daily_discharge.csv").open(encoding="utf-8-sig",newline="") as src:
        for row in csv.DictReader(src):
            if row["station"]=="nanping" and row["date"]<=stop:
                rows.append((row["date"],float(row["river_discharge_m3s"])))
    df = pd.DataFrame(rows,columns=["date","q"])
    df.index = pd.DatetimeIndex(df.date)
    if df.index.has_duplicates or not len(df):
        raise ValueError("Invalid Nanping history")
    q = df.q.sort_index().asfreq("D")
    # Target day T is issued just after source day T-1's aggregates are available.
    state = pd.DataFrame({"q_previous":q,"change3":q-q.shift(3)})
    state.index += DAY
    return state


def groups(frame, cutoff):
    high = np.asarray(frame.q_previous >= cutoff,dtype=int)
    rising = np.asarray(frame.change3 > 0,dtype=int)
    return 2*high+rising


def fit_gate(frame):
    cutoff = float(frame.q_previous.median())
    group = groups(frame,cutoff)
    d = (frame.prior-frame.local).to_numpy()
    residual = (frame.target - .5*(frame.prior+frame.local)).to_numpy()
    shrink = PRIOR_DAYS*np.mean(d*d)
    weights, counts = [], []
    for regime in range(4):
        mask = group==regime
        denom = np.sum(d[mask]**2)+shrink
        shift = np.sum(d[mask]*residual[mask])/denom if denom>0 else 0.
        weights.append(float(np.clip(FIXED_WEIGHT+shift,.25,.75)))
        counts.append(int(mask.sum()))
    return {"q_median":cutoff,"prior_weights":weights,"training_counts":counts,
            "last_training_target":str(frame.index.max().date())}


def predict_gate(frame, config):
    if frame.index.min() <= pd.Timestamp(config["last_training_target"]):
        raise ValueError("Benchmark inference requires targets strictly after the gate training period")
    regimes = groups(frame,config["q_median"])
    w = np.asarray(config["prior_weights"])[regimes]
    return pd.DataFrame({"group":regimes,"weight_prior":w,
                         "prediction":w*frame.prior + (1-w)*frame.local},index=frame.index)


def summarize(frame):
    result = []
    peak = frame.target >= np.quantile(frame.target,.95)
    for name in ["fixed","prediction"]:
        e = frame[name]-frame.target
        result.append({"model":name,"n":len(e),"rmse":float(np.mean(e**2)**.5),
                       "mae":float(np.mean(abs(e))),"peak_rmse":float(np.mean(e[peak]**2)**.5)})
    return pd.DataFrame(result).set_index("model")


def interval(frame):
    """Paired circular block bootstrap. Retains local clusters of flood errors."""
    e = (frame[["fixed","prediction"]].sub(frame.target,axis="index")**2).to_numpy()
    block, n = 14, len(e)
    rng = np.random.default_rng(20261121)
    starts = rng.integers(n,size=(4000,int(np.ceil(n/block))))
    indices = ((starts[:,:,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
    sums = e[indices].sum(axis=1)
    return [float(v) for v in np.quantile(1-np.sqrt(sums[:,1]/sums[:,0]),[.025,.975])]


def prepare(data, oof):
    frame = pd.read_csv(oof,usecols=["target_date","target","prior","local"],
                        index_col="target_date",parse_dates=["target_date"])
    if not frame.index.is_unique or frame.index.max()>pd.Timestamp("2020-12-31"):
        raise ValueError("Only development OOF through 2020-12-31 may be used")
    frame = frame.loc["2019-01-01":"2020-12-31"].copy()
    expected = pd.date_range("2019-01-01","2020-12-31",freq="D")
    if not frame.index.equals(expected):
        raise ValueError("Expected a complete daily 2019–2020 OOF grid")
    frame = frame.join(stage_history(data,"2020-12-30"))
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite OOF or gate state")
    return frame


def validate(data, oof, out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    frame = prepare(data,oof)
    train,hold = frame.loc["2019"],frame.loc["2020"]
    config = fit_gate(train)
    hold = hold.join(predict_gate(hold,config))
    hold["fixed"] = .5*(hold.prior+hold.local)
    scores = summarize(hold)
    bounds = interval(hold)
    gain = float(1-scores.loc["prediction","rmse"]/scores.loc["fixed","rmse"])
    # An update should improve the primary metric, retain typical-day accuracy
    # and remain favorable when nearby days are resampled together.
    passed = bool(gain>=MIN_GAIN and scores.loc["prediction","mae"]<=scores.loc["fixed","mae"] and bounds[0]>0)
    decision = dict(train_2019=config,forward_test="2020",relative_rmse_gain=gain,
                    paired_block14_interval=bounds,gate_accepted=passed,
                    threshold=dict(rmse_gain_at_least=MIN_GAIN,mae_no_worse=True,lower_bound_positive=True),
                    selected="group_gate" if passed else "v4_fixed",
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    oof_sha256=hashlib.sha256(Path(oof).read_bytes()).hexdigest())
    if passed:
        decision["deployment_gate"] = fit_gate(frame)
    hold.to_csv(out/"gate_2020_forward.csv",index_label="target_date")
    scores.to_csv(out/"gate_2020_scores.csv")
    (out/"gate_decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    print(scores.to_string())
    print(json.dumps(decision,indent=2))


def locked_blend(data, out, frame):
    """Apply the locked decision to existing V4 branch forecasts (no target column needed)."""
    decision = json.loads((Path(out)/"gate_decision.json").read_text(encoding="utf-8"))
    if decision["code_sha256"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Gate source differs from evaluated code")
    if not isinstance(frame.index,pd.DatetimeIndex) or frame.index.min()<=pd.Timestamp("2020-12-31"):
        raise ValueError("Supply dated predictions strictly after gate selection")
    if not frame.index.is_unique or not np.isfinite(frame[["prior","local"]]).all().all():
        raise ValueError("Invalid branch forecasts")
    if not decision["gate_accepted"]:
        return ((frame.prior+frame.local)*.5).rename("prediction")
    states = stage_history(data,frame.index.max()-DAY)
    x = frame[["prior","local"]].join(states)
    if not np.isfinite(x).all().all():
        raise ValueError("Missing gate inputs")
    return predict_gate(x,decision["deployment_gate"]).prediction


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--oof",type=Path,default=V4)
    p.add_argument("--out",type=Path,default=ROOT/"gate_results")
    args=p.parse_args()
    validate(args.data,args.oof,args.out)
