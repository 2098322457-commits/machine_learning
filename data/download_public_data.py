# -*- coding: utf-8 -*-
"""下载闽江流域公开水文气象数据，用于机器学习课程大作业（赛题：闽江流域水情预报）。

数据源（均为公开、无需账号的接口）：
1. Open-Meteo Historical Weather API（ERA5 / ERA5-Land 再分析）
   https://archive-api.open-meteo.com/v1/archive
   逐时：降水、2m 气温、相对湿度、地面气压、10m 风速
2. Open-Meteo Flood API（GloFAS v4 河流流量）
   https://flood-api.open-meteo.com/v1/flood
   日值：河流流量 river_discharge（m³/s），用作入库流量的代理目标
3. NASA POWER 日值（降水、气温），作为独立交叉校验源
   https://power.larc.nasa.gov/api/temporal/daily/point

用法：
    python download_public_data.py [--start 2014-01-01] [--end 2023-12-31]
输出写入本脚本所在目录。
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 闽江上游（水口水库以上）监测点近似坐标：气象站点 + 流量断面
STATIONS = [
    {"name": "wuyishan", "name_cn": "武夷山", "lat": 27.7566, "lon": 118.0353},
    {"name": "pucheng", "name_cn": "浦城", "lat": 27.9200, "lon": 118.5400},
    {"name": "jianyang", "name_cn": "建阳", "lat": 27.3300, "lon": 118.1200},
    {"name": "shaowu", "name_cn": "邵武", "lat": 27.3400, "lon": 117.4900},
    {"name": "guangze", "name_cn": "光泽", "lat": 27.5400, "lon": 117.3300},
    {"name": "jianou", "name_cn": "建瓯", "lat": 27.0300, "lon": 118.3200},
    {"name": "nanping", "name_cn": "南平", "lat": 26.6400, "lon": 118.1700},
    {"name": "sanming", "name_cn": "三明", "lat": 26.2600, "lon": 117.6400},
    {"name": "jiangle", "name_cn": "将乐", "lat": 26.7300, "lon": 117.4700},
    {"name": "taining", "name_cn": "泰宁", "lat": 26.9000, "lon": 117.1800},
    {"name": "youxi", "name_cn": "尤溪", "lat": 26.1700, "lon": 118.1900},
    {"name": "yongan", "name_cn": "永安", "lat": 25.9700, "lon": 117.3600},
]

FLOW_POINTS = [
    {"name": "shuikou", "name_cn": "水口坝址", "lat": 26.2167, "lon": 118.7333},
    {"name": "nanping", "name_cn": "南平断面", "lat": 26.6400, "lon": 118.1700},
]

HOURLY_VARS = [
    "precipitation",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


def get_json(url, tries=5, wait=10):
    """带重试的 GET，返回解析后的 JSON。"""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(wait * (i + 1))
                continue
            raise
        except Exception as e:  # 网络抖动
            last = e
            time.sleep(wait * (i + 1))
    raise RuntimeError("请求失败: %s (%s)" % (url, last))


def _fetch_station_hourly(st, start, end):
    """拉取单个站点的逐时气象要素。"""
    params = {
        "latitude": st["lat"],
        "longitude": st["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Shanghai",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    js = get_json(url)
    meta = {
        "station": st["name"],
        "station_cn": st["name_cn"],
        "latitude": js.get("latitude", st["lat"]),
        "longitude": js.get("longitude", st["lon"]),
        "elevation_m": js.get("elevation"),
    }
    print("  [open-meteo] %s %s 共 %d 小时" % (st["name"], st["name_cn"], len(js["hourly"]["time"])), flush=True)
    return st["name"], js["hourly"], meta


def download_openmeteo_hourly(start, end, workers=4):
    """逐站点拉取逐时气象要素，按变量分别保存为宽表（time × 站点）。"""
    store = {v: {} for v in HOURLY_VARS}
    meta = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_station_hourly, st, start, end) for st in STATIONS]
        for fut in concurrent.futures.as_completed(futures):
            name, hourly, m = fut.result()
            meta.append(m)
            for v in HOURLY_VARS:
                store[v][name] = pd.Series(hourly[v], index=hourly["time"])

    for v in HOURLY_VARS:
        df = pd.DataFrame(store[v])
        df.index.name = "time"
        path = os.path.join(OUT_DIR, "openmeteo_hourly_%s.csv" % v)
        df.to_csv(path, encoding="utf-8-sig")
        print("  写出 %s  %s" % (os.path.basename(path), df.shape), flush=True)

    pd.DataFrame(meta).sort_values("station").to_csv(
        os.path.join(OUT_DIR, "stations.csv"), index=False, encoding="utf-8-sig")


def download_glofas_daily(start, end):
    """GloFAS 日值河流流量，作为入库流量代理目标。"""
    params = {
        "latitude": ",".join(str(p["lat"]) for p in FLOW_POINTS),
        "longitude": ",".join(str(p["lon"]) for p in FLOW_POINTS),
        "daily": "river_discharge",
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Shanghai",
    }
    url = "https://flood-api.open-meteo.com/v1/flood?" + urllib.parse.urlencode(params)
    js = get_json(url)
    items = js if isinstance(js, list) else [js]
    frames = []
    for point, item in zip(FLOW_POINTS, items):
        d = item["daily"]
        frames.append(pd.DataFrame({
            "date": d["time"],
            "station": point["name"],
            "river_discharge_m3s": d["river_discharge"],
        }))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUT_DIR, "glofas_daily_discharge.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("  写出 %s  %s" % (os.path.basename(path), df.shape), flush=True)


def _fetch_nasa_power(st, s, e):
    """拉取单个站点的 NASA POWER 日值降水与气温。"""
    params = {
        "parameters": "PRECTOTCORR,T2M",
        "community": "AG",
        "longitude": st["lon"],
        "latitude": st["lat"],
        "start": s,
        "end": e,
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)
    js = get_json(url)
    p = js["properties"]["parameter"]
    dates = sorted(p["PRECTOTCORR"].keys())
    print("  [nasa-power] %s %s 共 %d 天" % (st["name"], st["name_cn"], len(dates)), flush=True)
    return pd.DataFrame({
        "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "station": st["name"],
        "precip_mm": [p["PRECTOTCORR"][d] for d in dates],
        "temp_c": [p["T2M"][d] for d in dates],
    })


def download_nasa_power_daily(start, end, workers=4):
    """NASA POWER 日值降水与气温，独立交叉校验源。"""
    s, e = start.replace("-", ""), end.replace("-", "")
    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_fetch_nasa_power, st, s, e) for st in STATIONS]
        for fut in concurrent.futures.as_completed(futures):
            frames.append(fut.result())
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUT_DIR, "nasa_power_daily.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("  写出 %s  %s" % (os.path.basename(path), df.shape), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    print("下载 Open-Meteo 逐时气象 ...", flush=True)
    download_openmeteo_hourly(args.start, args.end)
    print("下载 GloFAS 日值流量 ...", flush=True)
    download_glofas_daily(args.start, args.end)
    print("下载 NASA POWER 日值 ...", flush=True)
    download_nasa_power_daily(args.start, args.end)
    print("全部完成 ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
