#!/usr/bin/env python3
"""Cluster bootstrap (by variant) of the frozen contest statistics.
Requested at test-23b (in-house seat A): uncertainty for LOVO rel err
and dAIC with only 6 primary variants. Resamples variants with
replacement; per bootstrap, for each original variant v, models are
fit (weighted by bootstrap multiplicity) on reps of variants != v
and evaluated on v's reps (LOVO), and a weighted full fit gives dAIC.
Definitions identical to scripts/analyze.py. Banked data only.
Writes results/bootstrap_contest.json."""
import json, os
import numpy as np

rng = np.random.default_rng(20261026)
ROOT = os.path.join(os.path.dirname(__file__), "..")
ana = json.load(open(os.path.join(ROOT, "results/stage1_analysis.json")))
info = ana["variants"]; PRIMARY = ana["primary_variants"]
reps = []
for line in open(os.path.join(ROOT, "results/stage1_measurements.jsonl")):
    r = json.loads(line)
    if r["kind"] == "rep" and r.get("virgin") and not r.get("diagnostic"):
        reps.append(r)

def feats(v, ctx):
    S = ctx + 64; i = info[v]
    return {"P": i["P"] / 1e9, "S": S / 1e3, "L": i["L"],
            "B_W": i["B_W"] / 1e9, "B_KV": 4 * S * i["sum_kv"] * 128 / 1e9}

def design(rs, model):
    X, y = [], []
    for r in rs:
        f = feats(r["variant"], r["ctx"])
        X.append([1, f["P"], f["S"], f["P"] * f["S"], f["L"]] if model == "P"
                 else [1, f["B_W"], f["B_KV"], f["L"]])
        y.append(r["cpu_per_token"])
    return np.array(X), np.array(y)

def wfit(X, y, w):
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    pred = X @ b
    rss = float(np.sum(w * (y - pred) ** 2)); n = float(np.sum(w))
    return b, n * np.log(rss / n) + 2 * X.shape[1]

B = 2000
out = {}
for cap in (2, 4):
    data = [r for r in reps if r["cap"] == cap and r["ctx"] >= 4096
            and r["variant"] in PRIMARY]
    by_var = {v: [r for r in data if r["variant"] == v] for v in PRIMARY}
    d_aic, d_lovo = [], []
    for _ in range(B):
        draw = list(rng.choice(PRIMARY, size=len(PRIMARY), replace=True))
        mult = {v: draw.count(v) for v in PRIMARY}
        # dAIC from weighted full fit (variants absent get weight 0)
        aics = {}
        for model in ("P", "B"):
            X, y = design(data, model)
            w = np.array([mult[r["variant"]] for r in data], dtype=float)
            keep = w > 0
            _, aics[model] = wfit(X[keep], y[keep], w[keep])
        d_aic.append(aics["P"] - aics["B"])
        # LOVO: evaluate each ORIGINAL variant with models fit on the
        # bootstrap-weighted reps of the other variants
        errs = {"P": [], "B": []}
        for v in PRIMARY:
            tr = [r for r in data if r["variant"] != v and mult[r["variant"]] > 0]
            if not tr: continue
            wtr = np.array([mult[r["variant"]] for r in tr], dtype=float)
            te = by_var[v]
            for model in ("P", "B"):
                Xtr, ytr = design(tr, model); Xte, yte = design(te, model)
                sw = np.sqrt(wtr)
                b, *_ = np.linalg.lstsq(Xtr * sw[:, None], ytr * sw, rcond=None)
                errs[model] += list(np.abs(Xte @ b - yte) / yte)
        d_lovo.append(np.mean(errs["P"]) - np.mean(errs["B"]))
    d_aic = np.array(d_aic); d_lovo = np.array(d_lovo)
    out[str(cap)] = {
        "dAIC_median": float(np.median(d_aic)),
        "dAIC_ci95": [float(np.percentile(d_aic, 2.5)), float(np.percentile(d_aic, 97.5))],
        "P_dAIC_gt_10": float(np.mean(d_aic > 10)),
        "dLOVO_PminusB_median": float(np.median(d_lovo)),
        "dLOVO_ci95": [float(np.percentile(d_lovo, 2.5)), float(np.percentile(d_lovo, 97.5))],
        "n_boot": B}
json.dump(out, open(os.path.join(ROOT, "results/bootstrap_contest.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
