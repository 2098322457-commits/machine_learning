"""Daily flow change from a routing regression and a shallow boosted-tree correction."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent/"minjiang-original"/"data"
DAY = pd.Timedelta(days=1)
FIRST = pd.Timestamp("2014-02-01")
LAST_INNER = pd.Timestamp("2020-12-31")
BASINS = {"jianxi": ["jianou","jianyang","wuyishan","pucheng"],
          "futun": ["jiangle","guangze","shaowu","taining"],
          "shaxi": ["sanming","yongan","youxi"], "main": ["nanping"]}
# (fit a routing equation first, leaf limit, minimum leaf size, correction weight,
#  include yesterday's base-model error). Fixed before the outer and test periods.
OPTIONS = {
    "direct": (False, 12, 20, 1.0, False),
    "routing_smooth": (True, 5, 40, 0.6, False),
    "routing_feedback": (True, 7, 25, 0.8, True),
    "routing_feedback_deep": (True, 15, 20, 0.6, True),
}


def limited_read(path, last):
    # Both raw CSVs start with ISO dates. Values beyond the cutoff never reach the parser.
    bound = pd.Timestamp(last).date().isoformat().encode("ascii")
    with path.open("rb") as stream:
        header = stream.readline()
        selected = io.BytesIO(header + b"".join(row for row in stream if row[:10] <= bound))
    return pd.read_csv(selected, parse_dates=[0])


def tables(data, last, labelled):
    last = pd.Timestamp(last).normalize()
    data = Path(data)
    cutoff = last-DAY
    flow = limited_read(data/"glofas_daily_discharge.csv", last if labelled else cutoff)
    flow = flow.pivot(index="date", columns="station", values="river_discharge_m3s").sort_index().asfreq("D")
    labels = flow.nanping
    q = flow.loc[:cutoff]
    rain = limited_read(data/"openmeteo_hourly_precipitation.csv", cutoff)
    rain = rain.set_index("time", verify_integrity=True).sort_index().asfreq("h")
    level = q.nanping
    scale = 70 + .65*level + .35*level.rolling(5).mean()
    state = {"stage": np.log(scale), "antecedent": np.log((level.rolling(14).mean()+50)/(level+50))}
    gates = {"stage_gate": level/(level+900)}
    drivers = {}
    for site, s in q.items():
        delta = s.diff()
        ratio = s / (level+s+80)
        recent = s.rolling(5).mean()
        gates[site+"_rise"] = s/(s+recent+20)
        state[site+"_share"] = ratio
        state[site+"_rate"] = delta/(s+s.shift(1)+20)
        drivers[site+"_flow"] = s/scale
        for lag in range(4):
            drivers[f"{site}_dq{lag}"] = delta.shift(lag)/scale
    for basin, sites in BASINS.items():
        hours = rain[sites].mean(axis=1, skipna=False)
        daily = hours.resample("D").sum(min_count=24)
        wet = daily.rolling(10).mean().shift()
        state[basin+"_rain"] = np.log1p(daily)
        state[basin+"_wet"] = np.log1p(wet)
        for lag in range(3):
            drivers[f"{basin}_rain{lag}"] = daily.shift(lag)/scale
        for h in (0, 6, 12, 18):
            portion = hours.loc[hours.index.hour//6 == h//6].resample("D").sum(min_count=6)
            drivers[f"{basin}_h{h}"] = portion/scale
        drivers[basin+"_runoff"] = daily*wet/(4+wet)/scale
    drivers = pd.DataFrame(drivers)
    gate_drivers = drivers.filter(regex=r"(_flow|_dq0|_dq1|_rain0)$")
    # Smooth routing coefficients respond to tributary rising limbs and basin flow stage.
    interaction = [gate_drivers.mul(g, axis="index").add_prefix(name+"_")
                   for name, g in gates.items()]
    state = pd.DataFrame(state)
    linear = pd.concat([drivers, state, *interaction], axis=1)
    trees = pd.concat([drivers, state, pd.DataFrame(gates)], axis=1)
    available = linear.index[(linear.index + DAY >= FIRST) & (linear.index <= cutoff)]
    if available.empty or not available.equals(pd.date_range(FIRST-DAY, cutoff)):
        raise ValueError("Daily inputs do not cover all required issue days through "+str(cutoff.date()))
    idx = (available + DAY).rename("target_date")
    def align(x):
        value = x.loc[available].set_axis(idx)
        if not np.isfinite(value.to_numpy()).all():
            raise ValueError("Nonfinite features. Inspect flow and hourly rain coverage.")
        return value
    return align(linear), align(trees), align(level), align(scale), labels.reindex(idx)


def forecast_options(tables_, start, stop, names):
    linear, trees, level, scale, labels = tables_
    fit = (linear.index < start) & labels.notna()
    ev = (linear.index >= start) & (linear.index <= stop)
    requested = pd.date_range(start,stop)
    if not linear.index[ev].equals(requested) or fit.sum() < 365:
        raise ValueError("Incomplete output grid or insufficient training labels")
    if labels.loc[:stop-DAY].isna().any():
        raise ValueError("Missing past daily flow needed for error history")
    response = (labels-level)/scale
    w = (scale[fit]/scale[fit].mean())**1.5
    w /= w.mean()
    normalize = StandardScaler().fit(linear[fit], sample_weight=w)
    matrix = normalize.transform(linear)
    ridge = Ridge(alpha=15, solver="cholesky").fit(matrix[fit], response[fit], sample_weight=w)
    route = pd.Series(ridge.predict(matrix), index=linear.index)
    residual = response-route
    prior = residual.shift(1)
    base_columns = trees.assign(routing=route)
    feedback_columns = base_columns.assign(error1=prior.fillna(0), error2=prior.shift().fillna(0),
                                            recent_error=prior.ewm(halflife=2, adjust=False).mean().fillna(0))
    for name in names:
        anchored, leaves, min_leaf, correction, feedback = OPTIONS[name]
        x = (feedback_columns if feedback else base_columns) if anchored else trees
        y = residual if anchored else response
        model = HistGradientBoostingRegressor(max_iter=170, learning_rate=.045, max_leaf_nodes=leaves,
                    min_samples_leaf=min_leaf, l2_regularization=1.0, early_stopping=False, random_state=27)
        model.fit(x[fit], y[fit], sample_weight=w)
        base = route[ev] if anchored else 0
        change = base + correction*model.predict(x[ev])
        yield name, (level[ev]+scale[ev]*change).clip(lower=0).rename("prediction")


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate(data, artifacts):
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    table = tables(data, LAST_INNER, labelled=True)
    labels = table[-1]
    parts = {name: [] for name in OPTIONS}
    for year in (2018, 2019, 2020):
        start, stop = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        for name,pred in forecast_options(table, start, stop, OPTIONS):
            parts[name].append(pred)
    scores, years = {}, {}
    for name, pieces in parts.items():
        error = pd.concat(pieces) - labels.loc["2018":]
        if not np.isfinite(error).all() or len(error) != 1096:
            raise ValueError("Incomplete OOF candidate")
        scores[name] = float(np.sqrt(np.mean(error**2)))
        years[name] = {str(y):float(np.sqrt(np.mean(e**2))) for y,e in error.groupby(error.index.year)}
    winner = min(scores, key=scores.get)
    lock = dict(model=winner, last_selection_label=str(LAST_INNER.date()), source_sha256=source_hash(),
                n_oof=1096, oof_rmse=scores, yearly_rmse=years)
    choice = pd.concat(parts[winner])
    oof = pd.DataFrame({"target":labels.loc[choice.index], "prediction":choice})
    oof.to_csv(artifacts/"oof.csv", index_label="target_date")
    (artifacts/"choice.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


def predict(data, artifacts, start, stop):
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    lock = json.loads((Path(artifacts)/"choice.json").read_text(encoding="utf-8"))
    if start <= pd.Timestamp(lock["last_selection_label"]) or stop < start:
        raise ValueError("Requested dates precede the validation freeze or are out of order")
    if lock["source_sha256"] != source_hash():
        raise ValueError("Source changed since validation")
    table = tables(data, stop, labelled=False)
    with threadpool_limits(limits=2):
        return next(forecast_options(table, start, stop, [lock["model"]]))[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--artifacts", type=Path, default=ROOT/"trees")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        validate(args.data, args.artifacts)
