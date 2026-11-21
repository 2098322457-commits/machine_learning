"""Re-select the V3 daily benchmark using targets through 2020 only.

No precomputed V3 choices, OOF files or test scores are used. At each new
evaluation period we refit coefficients once. Hyperparameters stay frozen.
Targets are next completed calendar-day means for Nanping in the team dataset.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "minjiang-v3"
sys.path.insert(0, str(SOURCE))
try:
    import kernel_routing
    import routing
    import scale_routing
finally:
    sys.path.remove(str(SOURCE))


DAY = pd.Timedelta(days=1)
CUTOFF = pd.Timestamp("2020-12-31")
FOLDS = (2018, 2019, 2020)
BRANCHES = ("channel", "scale", "kernel")
WEATHER = ("precipitation", "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m")
FILES = ["glofas_daily_discharge.csv", "nasa_power_daily.csv"] + [
    f"openmeteo_hourly_{variable}.csv" for variable in WEATHER]


def limited_copy(data, destination, last):
    """Pass only eligible records to all V3 loaders, including their CSV parsers."""
    last = pd.Timestamp(last).date().isoformat().encode("ascii")
    for name in FILES:
        with (Path(data) / name).open("rb") as src, (destination / name).open("wb") as dst:
            header = src.readline()
            if header.split(b",")[0].removeprefix(b"\xef\xbb\xbf") not in (b"date", b"time"):
                raise ValueError(f"Unexpected leading timestamp column: {name}")
            dst.write(header)
            for line in src:
                if line[:10] <= last:
                    dst.write(line)


def code_hashes():
    files = [SOURCE / f"{name}.py" for name in ("routing", "scale_routing", "kernel_routing", "feedback")]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def rmse(y, pred):
    if not y.index.equals(pred.index):
        raise ValueError("OOF dates must match exactly")
    error = np.asarray(pred) - np.asarray(y)
    if not np.isfinite(error).all():
        raise ValueError("Invalid score values")
    return float(np.sqrt(np.mean(error ** 2)))


def fit_branches(data):
    xs, y = routing.build_tables(data, CUTOFF)
    index = pd.date_range("2018-01-01", CUTOFF, name="target_date")
    expected = y.loc[index]
    quality, winners, series = [], {}, {}

    def candidate(branch, params, pred):
        value = rmse(expected, pred)
        quality.append(dict(branch=branch, parameters=json.dumps(params), rmse=value))
        if branch not in winners or value < winners[branch]["rmse"]:
            winners[branch] = dict(rmse=value, parameters=params)
            series[branch] = pred.to_numpy()

    for family, x in xs.items():
        for alpha in (.03, .1, .3, 1., 3., 10., 30., 100., 300.):
            pieces = []
            for year in FOLDS:
                tr, ev = y.index.year < year, y.index.year == year
                p = routing.fit_predict(family, alpha, x[tr], y[tr], x[ev])
                pieces.append(pd.Series(p, index=y.index[ev]))
            candidate("channel", dict(family=family, alpha=alpha), pd.concat(pieces))
    del xs
    f, s, q = scale_routing.design(data, CUTOFF)
    label = q.shift(-1)
    dates = f.index + DAY
    ok = f.notna().all(axis=1) & label.notna() & (dates >= "2014-02-01")
    splits = [(ok & (dates.year < year), ok & (dates.year == year)) for year in FOLDS]
    for kind, design in [("linear", f), ("gated", scale_routing.response_basis(f))]:
        for alpha, exponent in itertools.product((3, 30, 120, 500, 1800), (.75, 1.5, 2.0)):
            pieces = [scale_routing.fit_predict(design, s, q, label, tr, ev, alpha, exponent)
                      for tr, ev in splits]
            pred = pd.concat(pieces)
            pred.index += DAY
            candidate("scale", dict(kind=kind, alpha=alpha, exponent=exponent), pred)
    options = list(itertools.product((1., 4., 16.), (.03, .3, 3.)))
    preds = {opt: [] for opt in options}
    for tr, ev in splits:
        for opt, pred in kernel_routing.predict_fold(f, s, q, label, tr, ev, options):
            preds[opt].append(pred)
    for (bandwidth, penalty), pieces in preds.items():
        pred = pd.concat(pieces)
        pred.index += DAY
        candidate("kernel", dict(bandwidth=bandwidth, penalty=penalty), pred)
    result = pd.DataFrame(series, index=index)
    result["target"] = expected
    return result, {k: v["parameters"] for k, v in winners.items()}, quality


def calibrate(history, period, alpha):
    """Use only earlier OOF residuals to fit and only already released observations as lags."""
    combined = pd.concat([history[["target", "base"]], period[["target", "base"]]])
    dates = pd.date_range(combined.index.min(), combined.index.max(), freq="D")
    if not combined.index.equals(dates):
        raise ValueError("Calibration archive must be contiguous and nonoverlapping")
    if alpha == "none":
        return period.base
    if combined.target.iloc[:-1].isna().any():
        raise ValueError("Feedback needs all targets strictly before the final prediction day")
    # The V3 feedback design, kept here so importing this benchmark never
    # imports V3's CLI module called 'run' into the V4 caller's namespace.
    prior = combined.target.shift(1)
    residual = (combined.target - combined.base).shift(1)
    rise = ((combined.base - prior) / (prior.abs() + 100)).clip(-1, 1)
    x = pd.DataFrame({"e1": residual, "e2": residual.shift(1), "e3": residual.shift(2),
                      "avg3": residual.rolling(3).mean(), "avg7": residual.rolling(7).mean(),
                      "e1_rising": residual * rise, "e2_rising": residual.shift(1) * rise,
                      "signed_e1": residual.abs() * rise,
                      "e1_stage": residual * np.log1p(prior / 500)}).fillna(0)
    scaler = StandardScaler().fit(x.loc[history.index])
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(x.loc[history.index]),
                                          history.target - history.base)
    correction = model.predict(scaler.transform(x.loc[period.index]))
    return (period.base + correction).clip(lower=0)


def freeze(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="v3-prior-2020-") as temp, threadpool_limits(limits=2):
        clipped = Path(temp)
        limited_copy(data, clipped, CUTOFF)
        hist, params, log = fit_branches(clipped)
    best_score = np.inf
    for i in range(5):
        for j in range(5-i):
            weights = np.array([i, j, 4-i-j]) / 4
            pred = hist[list(BRANCHES)] @ weights
            score = rmse(hist.target, pred)
            log.append(dict(branch="blend", parameters=json.dumps(weights.tolist()), rmse=score))
            if score < best_score:
                best_score = score
                best_weights = dict(zip(BRANCHES, weights.tolist(), strict=True))
                hist["base"] = pred
    best_score = np.inf
    for alpha in ("none", "30", "300", "1500"):
        segments = [calibrate(hist[hist.index.year < year], hist[hist.index.year == year], alpha)
                    for year in (2019, 2020)]
        pred = pd.concat(segments)
        score = rmse(hist.target.loc[pred.index], pred)
        log.append(dict(branch="feedback", parameters=alpha, rmse=score))
        if score < best_score:
            best_score, strength = score, alpha
            hist["prior"] = pred
    hist.to_csv(out / "prior_oof.csv")
    pd.DataFrame(log).to_csv(out / "prior_validation.csv", index=False)
    lock = dict(last_selection_label=str(CUTOFF.date()), branch_parameters=params, weights=best_weights,
                calibration_strength=strength, source_sha256=code_hashes(),
                oof_sha256=hashlib.sha256((out / "prior_oof.csv").read_bytes()).hexdigest())
    (out / "prior_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def forecast_segment(data, start, stop, lock):
    """Fit each nonzero branch once at period start, then use daily observations through t-1."""
    horizon = start - DAY
    expected = pd.date_range(start, stop, name="target_date")
    xs, y = routing.build_tables(data, stop, require_labels=False)
    base = pd.Series(0., index=expected, name="base")
    weights = lock["weights"]
    if weights["channel"]:
        p = lock["branch_parameters"]["channel"]
        x = xs[p["family"]]
        tr = (y.index <= horizon) & y.notna()
        base += weights["channel"] * routing.fit_predict(p["family"], p["alpha"], x[tr], y[tr], x.loc[expected])
    del xs
    if weights["scale"] or weights["kernel"]:
        f, s, q = scale_routing.design(data, stop)
        label = q.shift(-1)
        dates = f.index + DAY
        ok = f.notna().all(axis=1)
        tr = ok & label.notna() & (dates >= "2014-02-01") & (dates <= horizon)
        ev = dates.isin(expected)
        if not ok[ev].all() or not dates[ev].equals(expected):
            raise ValueError("Incomplete feature dates")
        if weights["scale"]:
            p = lock["branch_parameters"]["scale"]
            design = scale_routing.response_basis(f) if p["kind"] == "gated" else f
            pred = scale_routing.fit_predict(design, s, q, label, tr, ev, p["alpha"], p["exponent"])
            base += weights["scale"] * pred.to_numpy()
        if weights["kernel"]:
            p = lock["branch_parameters"]["kernel"]
            _, pred = next(kernel_routing.predict_fold(f, s, q, label, tr, ev, [(p["bandwidth"], p["penalty"])]))
            base += weights["kernel"] * pred.to_numpy()
    return pd.DataFrame({"base": base, "target": y.loc[expected]})


def predict(data, out, start, stop):
    """Start in 2021 or 2022. Fixed fit for each segment, including the 2-year 2022–23 segment."""
    out = Path(out)
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if start not in (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")) or stop < start:
        raise ValueError("Supported period starts: 2021-01-01, 2022-01-01")
    lock = json.loads((out / "prior_lock.json").read_text(encoding="utf-8"))
    if lock["source_sha256"] != code_hashes() or lock["last_selection_label"] != "2020-12-31":
        raise ValueError("V3 sources or cutoff differ from the frozen selection")
    hist_file = out / "prior_oof.csv"
    if hashlib.sha256(hist_file.read_bytes()).hexdigest() != lock["oof_sha256"]:
        raise ValueError("Fitted calibration archive does not match freeze")
    history = pd.read_csv(hist_file, index_col="target_date", parse_dates=True)
    with threadpool_limits(limits=2):
        if start.year == 2022:
            bridge = forecast_segment(data, pd.Timestamp("2021-01-01"), start - DAY, lock)
            history = pd.concat([history[["target","base"]], bridge])
        period = forecast_segment(data, start, stop, lock)
        period["prior"] = calibrate(history, period, lock["calibration_strength"])
    return period[["target", "base", "prior"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "predict"])
    parser.add_argument("--data", type=Path, default=routing.DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "prior_results")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--stop", default="2021-12-31")
    args = parser.parse_args()
    if args.stage == "freeze":
        print(json.dumps(freeze(args.data, args.out), indent=2))
    else:
        predict(args.data, args.out, args.start, args.stop).to_csv(args.out / f"prior_{args.start}_{args.stop}.csv")
