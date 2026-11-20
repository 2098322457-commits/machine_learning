# -*- coding: utf-8 -*-
"""由 data/ 下的公开气象水文数据构建日尺度建模特征表。

预测目标：次日（24 小时后）南平断面日均流量。

输出：outputs/model_table_daily.csv
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")

# 子流域划分（按站点所在河流）
SUB_BASINS = {
    "jianxi": ["jianyang", "jianou", "pucheng", "wuyishan"],     # 建溪
    "futunxi": ["shaowu", "guangze", "taining", "jiangle"],      # 富屯溪
    "shaxi": ["sanming", "yongan", "youxi"],                     # 沙溪
    "ganliu": ["nanping"],                                       # 干流区间
}
TARGET_STATION = "nanping"
LAGS = [1, 2, 3, 4, 5, 6, 7]
WINDOWS = [3, 7, 30]


def load_hourly(name):
    df = pd.read_csv(os.path.join(DATA_DIR, "openmeteo_hourly_%s.csv" % name), parse_dates=["time"])
    return df.set_index("time")


def rain_features():
    """分区面雨量：日累计 + 滞后 + 滑动累积。"""
    rain = load_hourly("precipitation")
    daily = rain.resample("D").sum()

    groups = {}
    for basin, cols in SUB_BASINS.items():
        groups[basin] = daily[cols].mean(axis=1)
    groups["basin"] = daily.mean(axis=1)

    feats = {}
    for name, s in groups.items():
        feats["rain_%s" % name] = s
        for lag in LAGS:
            feats["rain_%s_lag%d" % (name, lag)] = s.shift(lag)
        for w in WINDOWS:
            feats["rain_%s_sum%d" % (name, w)] = s.rolling(w).sum()
            feats["rain_%s_sum%d_lag1" % (name, w)] = s.rolling(w).sum().shift(1)
    return pd.DataFrame(feats)


def meteo_features():
    """气温、湿度、气压、风速的流域日均值及一阶滞后。"""
    feats = {}
    for var, agg in [("temperature_2m", "mean"), ("relative_humidity_2m", "mean"),
                     ("surface_pressure", "mean"), ("wind_speed_10m", "mean")]:
        s = getattr(load_hourly(var).resample("D"), agg)().mean(axis=1)
        short = var.split("_")[0]
        feats["met_%s" % short] = s
        feats["met_%s_lag1" % short] = s.shift(1)
    return pd.DataFrame(feats)


def flow_features():
    """干支流各断面流量（当日值即昨日观测，可用于预报次日）。"""
    q = pd.read_csv(os.path.join(DATA_DIR, "glofas_daily_discharge.csv"), parse_dates=["date"])
    wide = q.pivot_table(index="date", columns="station", values="river_discharge_m3s").sort_index()
    feats = {}
    for st in wide.columns:
        for lag in [0, 1, 2, 3]:
            feats["q_%s_lag%d" % (st, lag)] = wide[st].shift(lag)
    return pd.DataFrame(feats), wide


def build_table():
    rain = rain_features()
    meteo = meteo_features()
    flow, wide = flow_features()

    df = pd.concat([rain, meteo, flow], axis=1)
    df.index.name = "date"

    # 日历特征
    df["doy_sin"] = np.sin(2 * np.pi * df.index.day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.day_of_year / 365.25)
    df["month"] = df.index.month
    df["is_flood_season"] = df["month"].isin([4, 5, 6, 7, 8, 9]).astype(int)

    # 目标：次日南平断面日均流量
    df["y"] = wide[TARGET_STATION].shift(-1)

    df = df.dropna()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_table()
    path = os.path.join(OUT_DIR, "model_table_daily.csv")
    df.to_csv(path, encoding="utf-8-sig")
    print("特征表:", df.shape, "->", path)
    print("时间范围:", df.index.min().date(), "->", df.index.max().date())
    print("目标统计: mean=%.1f  p95=%.1f  max=%.1f"
          % (df["y"].mean(), df["y"].quantile(0.95), df["y"].max()))


if __name__ == "__main__":
    main()
