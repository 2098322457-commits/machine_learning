# -*- coding: utf-8 -*-
"""三种方法的日尺度流量预测实验。

切分：训练 2014-2019，验证 2020-2021（选超参），测试 2022-2023。
方法：Ridge 回归、随机森林、直方图梯度提升（HistGradientBoosting）。
基线：持续性（昨日流量）、训练集均值。

输出：outputs/metrics.csv, outputs/predictions_test.csv, outputs/feature_importance.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
TABLE = os.path.join(OUT_DIR, "model_table_daily.csv")

TRAIN_YEARS = range(2014, 2020)
VALID_YEARS = range(2020, 2022)
TEST_YEARS = range(2022, 2024)


def metrics(y, pred, label):
    y, pred = np.asarray(y, float), np.clip(np.asarray(pred, float), 0, None)
    peak = y >= np.quantile(y, 0.95)
    return {
        "model": label,
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
        "SMAPE": float(np.mean(2 * np.abs(pred - y) / (np.abs(pred) + np.abs(y) + 1e-6)) * 100),
        "peak_RMSE": mean_squared_error(y[peak], pred[peak]) ** 0.5,
    }


def main():
    df = pd.read_csv(TABLE, parse_dates=["date"]).set_index("date")
    feat_cols = [c for c in df.columns if c != "y"]

    train = df[df.index.year.isin(TRAIN_YEARS)]
    valid = df[df.index.year.isin(VALID_YEARS)]
    test = df[df.index.year.isin(TEST_YEARS)]
    print("样本量  训练 %d | 验证 %d | 测试 %d" % (len(train), len(valid), len(test)))

    Xtr, ytr = train[feat_cols], train["y"]
    Xva, yva = valid[feat_cols], valid["y"]
    Xte, yte = test[feat_cols], test["y"]

    # 1. 线性回归：在验证集上选正则强度
    best_alpha, best_rmse = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr, ytr)
        rmse = mean_squared_error(yva, np.clip(model.predict(Xva), 0, None)) ** 0.5
        print("  Ridge alpha=%-6s 验证 RMSE=%.1f" % (alpha, rmse))
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(
        pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))

    # 2. 随机森林
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.3,
                               n_jobs=-1, random_state=0).fit(Xtr, ytr)

    # 3. 梯度提升树
    hgb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, min_samples_leaf=20,
                                        l2_regularization=1.0, random_state=0).fit(Xtr, ytr)

    preds = {
        "持续性基线": test["q_nanping_lag0"].values,
        "训练均值基线": np.full(len(test), ytr.mean()),
        "Ridge(alpha=%g)" % best_alpha: ridge.predict(Xte),
        "随机森林": rf.predict(Xte),
        "梯度提升树": hgb.predict(Xte),
    }
    preds["等权集成"] = np.mean([preds["Ridge(alpha=%g)" % best_alpha],
                                 preds["随机森林"], preds["梯度提升树"]], axis=0)

    rows = [metrics(yte, p, name) for name, p in preds.items()]
    table = pd.DataFrame(rows).set_index("model").round(3)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "metrics.csv"), encoding="utf-8-sig")
    print("\n测试集（2022-2023）指标：")
    print(table.to_string())

    out = pd.DataFrame({"y": yte})
    for name, p in preds.items():
        out[name] = np.clip(p, 0, None)
    out.to_csv(os.path.join(OUT_DIR, "predictions_test.csv"), encoding="utf-8-sig")

    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    imp.head(20).round(4).to_csv(os.path.join(OUT_DIR, "feature_importance.csv"),
                                 encoding="utf-8-sig", header=["importance"])
    print("\n随机森林 Top10 重要特征：")
    print(imp.head(10).round(4).to_string())

    plot_predictions(out, best_alpha)


def plot_predictions(out, best_alpha):
    """测试期预测曲线（含 2022 年 6 月洪水）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(out.index, out["y"], color="black", lw=1.2, label="实际流量")
    ax.plot(out.index, out["Ridge(alpha=%g)" % best_alpha], color="#1f77b4", lw=1.0, label="Ridge")
    ax.plot(out.index, out["梯度提升树"], color="#ff7f0e", lw=0.8, alpha=0.8, label="梯度提升树")
    ax.set_xlabel("日期")
    ax.set_ylabel("日均流量 (m³/s)")
    ax.set_title("测试集（2022-2023）次日流量预测")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "pred_vs_actual.png")
    fig.savefig(path, dpi=150)
    print("\n预测曲线 ->", path)


if __name__ == "__main__":
    main()
