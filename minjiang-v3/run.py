"""Reproduce, audit or use the daily Nanping forecast with the group data directory."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import kernel_routing
import routing
import scale_routing


ROOT = Path(__file__).resolve().parent
DAY = pd.Timedelta(days=1)
MODELS = {"channel":routing, "scale":scale_routing, "kernel":kernel_routing}


def model_dir(out, name):
    return out/("routing_results" if name=="channel" else "experiments")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_signatures(out):
    files = [ROOT/(m.__name__+".py") for m in MODELS.values()] + [
        out/"routing_results/lock.json",out/"experiments/scale_choice.json",out/"experiments/kernel_choice.json"]
    return {p.name:sha(p) for p in files}


def metrics(actual,pred):
    e = np.asarray(pred)-np.asarray(actual)
    if not len(e) or not np.isfinite(e).all():
        raise ValueError("Incomplete target or forecast")
    high = np.asarray(actual)>=np.quantile(actual,.95)
    return dict(n=len(e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),
                peak_rmse=float(np.sqrt(np.mean(e[high]**2))))


def select(out):
    files = [out/"routing_results/oof.csv",out/"experiments/scale_oof.csv",out/"experiments/kernel_oof.csv"]
    dates = pd.date_range("2018","2021-12-31",name="target_date")
    preds = {}
    labels = None
    for name,path in zip(MODELS,files,strict=True):
        d = pd.read_csv(path,index_col="target_date",parse_dates=True)
        if not d.index.equals(dates):
            raise ValueError(f"Validation dates differ in {path}")
        if labels is not None:
            np.testing.assert_allclose(d.y,labels)
        labels = d.y
        preds[name] = d.prediction
    rows = []
    for i in range(5):
        for j in range(5-i):
            weights = dict(zip(MODELS,[i/4,j/4,(4-i-j)/4],strict=True))
            pred = sum(weights[name]*series for name,series in preds.items())
            rows.append(dict(**weights,**metrics(labels,pred)))
    grid = pd.DataFrame(rows).sort_values("rmse")
    grid.to_csv(out/"blend_validation.csv",index=False)
    weights = {name:float(grid.iloc[0][name]) for name in MODELS}
    choice = dict(weights=weights, pooled_oof_rmse=float(grid.iloc[0].rmse),
                  last_selection_date="2021-12-31",signatures=model_signatures(out))
    (out/"frozen.json").write_text(json.dumps(choice,indent=2),encoding="utf-8")
    print("Selected on 2018–2021:",weights,choice["pooled_oof_rmse"])


def forecast(data,out,start,stop,delay):
    lock = json.loads((out/"frozen.json").read_text(encoding="utf-8"))
    if model_signatures(out)!=lock["signatures"]:
        raise ValueError("Component version changed. Run validation again.")
    if delay<0 or start>stop or pd.Timestamp(start)-delay*DAY <= pd.Timestamp(lock["last_selection_date"]):
        raise ValueError("Requested range starts before the frozen selection is available")
    expected = pd.date_range(start,stop,name="target_date")
    result = pd.DataFrame(index=expected)
    for name,weight in lock["weights"].items():
        if weight:
            p = MODELS[name].locked_predict(data,model_dir(out,name),start,stop,delay)
            if not p.index.equals(expected) or not np.isfinite(p).all():
                raise ValueError(f"Incomplete predictions from {name}")
            result[name] = p
    result["v3"] = sum(lock["weights"][name]*result[name] for name in result)
    return result


def compare(data,out):
    rdir = model_dir(out,"channel")
    routing.evaluate(SimpleNamespace(data=data,out=rdir))
    baseline = pd.read_csv(rdir/"predictions.csv",index_col="target_date",parse_dates=True)
    result = forecast(data,out,"2022-01-01","2023-12-31",0)
    q = pd.read_csv(data/"glofas_daily_discharge.csv",parse_dates=["date"])
    truth = q[q.station=="nanping"].set_index("date").river_discharge_m3s
    result["target"] = truth.loc[result.index]
    np.testing.assert_allclose(result.target,baseline.y)
    result = result.join(baseline[["persistence","group_old_fixed","v2_old_fixed"]])
    result.to_csv(out/"test_predictions.csv")
    rows = []
    for name in result.columns.drop("target"):
        for span in ("all","2022","2023"):
            frame = result if span=="all" else result.loc[span]
            rows.append(dict(model=name,period=span,**metrics(frame.target,frame[name])))
    scores = pd.DataFrame(rows)
    scores.to_csv(out/"test_metrics.csv",index=False)
    print(scores.loc[scores.period=="all"].to_string(index=False))
    # Selection used observations through Dec 31. With one extra day of delay,
    # the first target that can use this frozen selection is January 2.
    late = forecast(data,out,"2022-01-02","2023-12-31",1)
    y = truth.loc[late.index]
    stale = truth.reindex(late.index-2*DAY).to_numpy()
    timely = result.loc[late.index]
    delay_table = pd.DataFrame([dict(scenario=name,**metrics(y,p)) for name,p in
             [("normal_same_dates",timely.v3),("one_day_late_refit",late.v3),("late_persistence",stale)]])
    delay_table.to_csv(out/"delay_metrics.csv",index=False)
    late.to_csv(out/"delay_predictions.csv")
    print(delay_table.to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["all","validate","select","evaluate","predict"])
    parser.add_argument("--data",type=Path,default=scale_routing.DATA)
    parser.add_argument("--out",type=Path,default=ROOT)
    parser.add_argument("--date",help="Target YYYY-MM-DD, for predict")
    parser.add_argument("--delay-days",type=int,default=0)
    args=parser.parse_args()
    if args.stage in ("all","validate"):
        rdir=model_dir(args.out,"channel")
        rdir.mkdir(exist_ok=True,parents=True)
        routing.validate(SimpleNamespace(data=args.data,out=rdir))
        scale_routing.validation(args.data,model_dir(args.out,"scale"))
        kernel_routing.validate(args.data,model_dir(args.out,"kernel"))
    if args.stage in ("all","validate","select"):
        select(args.out)
    if args.stage in ("all","evaluate"):
        compare(args.data,args.out)
    if args.stage=="predict":
        if not args.date:
            parser.error("--date is required")
        print(forecast(args.data,args.out,args.date,args.date,args.delay_days).to_csv(),end="")


if __name__=="__main__":
    with threadpool_limits(limits=3):
        main()
