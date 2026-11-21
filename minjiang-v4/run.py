"""Daily Nanping forecast with separate development, outer validation and test periods."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import candidate
import prior
import ar_model

ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=candidate.DATA
KEYS=("prior","local","ar")
SOURCE_DIRS=("prior_results","candidate_results","ar")
LOCKS=("prior_lock.json","candidate_choice.json","ar_lock.json")


def metrics(y,p):
    error=np.asarray(p)-np.asarray(y)
    if not np.isfinite(error).all() or not len(error):
        raise ValueError("Missing or nonfinite target/prediction")
    peak=y >= np.quantile(y,.95)
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.abs(error).mean()),
                peak_rmse=float(np.sqrt(np.mean(error[peak]**2))))


def signatures(root):
    paths=[root/d/name for d,name in zip(SOURCE_DIRS,LOCKS,strict=True)]
    paths += [ROOT/f for f in ("run.py","ar_model.py","prior.py","candidate.py")]
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def select(out,model_root):
    out.mkdir(parents=True,exist_ok=True)
    paths=[model_root/d/file for d,file in zip(SOURCE_DIRS,("prior_oof.csv","candidate_oof.csv","ar_oof.csv"))]
    df=pd.DataFrame(index=pd.date_range("2018-01-01","2020-12-31",name="target_date"))
    for name,path in zip(KEYS,paths,strict=True):
        p=pd.read_csv(path,index_col="target_date",parse_dates=True)
        if not df.index.equals(p.index):
            raise ValueError(f"Inner dates differ: {name}")
        if name=="prior":
            df["target"]=p.target
            df["prior"]=p.prior
        else:
            np.testing.assert_allclose(df.target,p.y)
            df[name]=p.prediction
    scored=df.loc["2019":"2020"]
    rows=[]
    for i in range(5):
        for j in range(5-i):
            weights=dict(zip(KEYS,(i/4,j/4,(4-i-j)/4),strict=True))
            pred=scored[list(KEYS)]@np.array(list(weights.values()))
            rows.append(dict(**weights,components=sum(w>0 for w in weights.values()),**metrics(scored.target,pred)))
    grid=pd.DataFrame(rows).sort_values("rmse")
    # Prefer the simpler blend when its pooled inner RMSE is within 0.5%.
    near=grid.loc[grid.rmse<=1.005*grid.rmse.min()]
    best=near.sort_values(["components","rmse"]).iloc[0]
    weights={key:float(best[key]) for key in KEYS}
    lock=dict(last_selection_label="2020-12-31",weights=weights,inner_n=len(scored),
              inner_rmse=float(best.rmse),component_signatures=signatures(model_root))
    (out/"frozen.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    grid.to_csv(out/"blend_candidates.csv",index=False)
    df.loc[scored.index,"prediction"]=scored[list(KEYS)]@np.array(list(weights.values()))
    df.to_csv(out/"validation_oof.csv")
    print("Frozen on 2019–2020 inner validation:",weights,round(best.rmse,5))


def frozen_predict(data_dir,out_dir,start,stop):
    out=Path(out_dir)
    root=out.parent
    lock=json.loads((out/"frozen.json").read_text(encoding="utf-8"))
    if lock["component_signatures"]!=signatures(root):
        raise ValueError("Model source or configuration changed since selection")
    start,stop=pd.Timestamp(start),pd.Timestamp(stop)
    if start not in (pd.Timestamp("2021-01-01"),pd.Timestamp("2022-01-01")) or stop<start:
        raise ValueError("Use a locked benchmark start (2021-01-01 or 2022-01-01)")
    frame=prior.predict(data_dir,root/SOURCE_DIRS[0],start,stop)[["prior"]].copy()
    frame["local"]=candidate.locked_predict(data_dir,root/SOURCE_DIRS[1],start,stop)
    frame["ar"]=ar_model.locked_predict(data_dir,root/SOURCE_DIRS[2],start,stop)
    if not frame.index.equals(pd.date_range(start,stop)) or not np.isfinite(frame).all().all():
        raise ValueError("Incomplete branch forecast")
    frame["prediction"]=frame[list(KEYS)] @ np.array([lock["weights"][k] for k in KEYS])
    return frame.rename_axis("target_date")


def score_period(data,out,start,stop,period):
    frame=frozen_predict(data,out,start,stop)
    tables,y=ar_model.build_tables(data,stop)
    group=tables["group"]
    before=group.index<pd.Timestamp(start)
    ev=group.index.isin(frame.index)
    frame["group"]=prior.routing.fit_predict("group",.1,group[before],y[before],group[ev])
    frame["persistence"]=group.loc[frame.index,"q_nanping_lag0"]
    frame.insert(0,"target",y.reindex(frame.index))
    frame.to_csv(out/(period+".csv"))
    return frame


def evaluate(data,out):
    inner=pd.read_csv(out/"validation_oof.csv",index_col="target_date",parse_dates=True).loc["2019":"2020"]
    outer=score_period(data,out,"2021-01-01","2021-12-31","outer_2021")
    test=score_period(data,out,"2022-01-01","2023-12-31","test_2022_23")
    rows=[]
    for period,frame in [("inner_2019_20",inner),("outer_2021",outer),("test_2022_23",test),
                         ("test_2022",test.loc["2022"]),("test_2023",test.loc["2023"])]:
        for name in frame.columns.drop("target"):
            rows.append(dict(period=period,model=name,**metrics(frame.target,frame[name])))
    table=pd.DataFrame(rows)
    table.to_csv(out/"metrics.csv",index=False)
    print(table.loc[table.model.isin(["prior","prediction","group"])].to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("all","validate","select","evaluate","predict"))
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--work",type=Path,default=ROOT,help="Artifact root; final scores appear in results/")
    parser.add_argument("--date",help="Target date for single-day prediction")
    args=parser.parse_args()
    root=args.work.resolve()
    out=root/"results"
    if args.stage in ("all","validate"):
        prior.freeze(args.data,root/SOURCE_DIRS[0])
        candidate.validate(args.data,root/SOURCE_DIRS[1])
        ar_model.validate(args.data,root/SOURCE_DIRS[2])
    if args.stage in ("all","validate","select"):
        select(out,root)
    if args.stage in ("all","evaluate"):
        evaluate(args.data,out)
    if args.stage=="predict":
        if not args.date or pd.Timestamp(args.date)<pd.Timestamp("2021-01-01"):
            parser.error("Choose --date on or after 2021-01-01")
        start="2021-01-01" if pd.Timestamp(args.date).year==2021 else "2022-01-01"
        print(frozen_predict(args.data,out,start,args.date).prediction.tail(1).to_csv(),end="")


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
