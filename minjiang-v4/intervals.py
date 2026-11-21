"""Paired circular 14-day blocks for the RMSE difference on identical target days."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT/"results"
    old = pd.read_csv(ROOT.parent/"minjiang-v3/feedback_predictions.csv",index_col="target_date")
    rng = np.random.default_rng(2026)
    rows = []
    for phase in ("outer_2021","test_2022_23"):
        d = pd.read_csv(out/(phase+".csv"),index_col="target_date")
        references = ["group","prior"]
        if phase=="test_2022_23":
            d["published_v3"]=old.loc[d.index,"v3_feedback"]
            references.append("published_v3")
        for ref in references:
            errors=d[["prediction",ref]].sub(d.target,axis=0).pow(2).to_numpy()
            starts=rng.integers(0,len(d),size=(2000,(len(d)+13)//14,1))
            idx=((starts+np.arange(14))%len(d)).reshape(2000,-1)[:,:len(d)]
            rmse=np.sqrt(errors[idx].mean(axis=1))
            lo,hi=np.quantile(rmse[:,1]-rmse[:,0],[.025,.975])
            rows.append(dict(phase=phase,reference=ref,rmse_gain_low95=lo,rmse_gain_high95=hi))
    pd.DataFrame(rows).to_csv(out/"block_intervals.csv",index=False)


if __name__=="__main__":
    main()
