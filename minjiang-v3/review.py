"""Independent scores, year/flow diagnostics and paired uncertainty from daily output."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from scale_routing import ROOT, DATA
from run import metrics


def review(data,out):
    forecast=pd.read_csv(out/"feedback_predictions.csv",index_col="target_date",parse_dates=True)
    raw=pd.read_csv(data/"glofas_daily_discharge.csv",parse_dates=["date"])
    y=raw.loc[raw.station=="nanping"].set_index("date").river_discharge_m3s.reindex(forecast.index)
    if not forecast.index.equals(pd.date_range("2022-01-01","2023-12-31")) or y.isna().any():
        raise ValueError("Dates or labels differ from the benchmark grid")
    scores=[]
    for year in (2022,2023,"all"):
        keep = np.ones(len(y),dtype=bool) if year=="all" else (y.index.year == year)
        for name,p in forecast.items():
            scores.append({"period":year,"model":name,**metrics(y[keep],p[keep])})
    pd.DataFrame(scores).to_csv(out/"final_metrics.csv",index=False)
    # Paired moving blocks preserve short-run dependence within weather events.
    rng=np.random.default_rng(2026)
    sq=(forecast.sub(y,axis=0))**2
    sample=[]
    for _ in range(2000):
        starts=rng.integers(0,len(y),size=(len(y)+13)//14)
        idx=((starts[:,None]+np.arange(14))%len(y)).ravel()[:len(y)]
        sample.append(np.sqrt(sq.to_numpy()[idx].mean(axis=0)))
    boot=pd.DataFrame(sample,columns=forecast.columns)
    dif=boot.subtract(boot["v3_feedback"],axis=0)
    ci=dif.quantile([.025,.975]).T.rename(columns={.025:"improvement_low95",.975:"improvement_high95"})
    ci.to_csv(out/"paired_14day.csv")
    p=pd.DataFrame({"target":y,"group":forecast.group,"v3":forecast.v3_feedback})
    t=y.idxmax()
    segment=p.loc[t-pd.Timedelta(days=9):t+pd.Timedelta(days=9)]
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.spines.top":False,
                         "axes.spines.right":False,"axes.unicode_minus":False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(11.6,4.2),layout="constrained",gridspec_kw={"width_ratios":[1.4,1]})
    for name,label,color in [("target","南平日流量标签","#303f4f"),("group","组员 Ridge","#ba772a"),
                             ("v3","第三版","#047c85")]:
        ax.plot(segment.index,segment[name],label=label,color=color,lw=1.8)
    ax.set(title="测试期最大流量前后九天",ylabel="日均流量（m³/s）",xlabel=f"{t.year} 年")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    rmse=pd.DataFrame(scores).query("period == 'all'").set_index("model").rmse
    names=["group","v2","v3","v3_feedback"]
    bars=bx.barh(["组员原版","第二版","第三版固定组合","第三版加误差校正"],rmse[names],
                 color=["#aab5bc","#859fa4","#448b90","#047c85"])
    bx.invert_yaxis()
    bx.set(title="同一批 730 个目标日",xlabel="RMSE（m³/s），越低越好",xlim=(0,140))
    bx.bar_label(bars,fmt="%.2f",padding=3)
    for axis in (ax,bx):
        axis.grid(axis="y" if axis is ax else "x",color="#dde3e6")
        axis.set_axisbelow(True)
    fig.savefig(out/"daily_review.png",dpi=150)
    plt.close(fig)
    print(ci.round(2).to_string())
    print("Largest target:",t.date(),p.loc[t].round(2).to_dict())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=ROOT)
    args=parser.parse_args()
    review(args.data,args.out)
