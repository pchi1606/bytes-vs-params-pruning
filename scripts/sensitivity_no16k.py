#!/usr/bin/env python3
"""Post-hoc sensitivity (requested at test-23b R3, NOT part of the frozen
criteria): rerun the frozen contest definitions (same design matrices,
LOVO, AIC as scripts/analyze.py) excluding the saturated 16384 context.
First replicates the frozen C>=4096 contest as a sanity check.
Reads results/stage1_measurements.jsonl + results/stage1_analysis.json
(variant info). Writes results/sensitivity_no16k.json."""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
ana = json.load(open(os.path.join(ROOT, "results/stage1_analysis.json")))
info = ana["variants"]
PRIMARY = [v for v in ana["primary_variants"]]

reps = []
for line in open(os.path.join(ROOT, "results/stage1_measurements.jsonl")):
    r = json.loads(line)
    if r["kind"] == "rep" and r.get("virgin") and not r.get("diagnostic"):
        reps.append(r)

def feats(v, ctx):
    S = ctx + 64
    i = info[v]
    return {"P": i["P"] / 1e9, "S": S / 1e3, "L": i["L"],
            "B_W": i["B_W"] / 1e9, "B_KV": 4 * S * i["sum_kv"] * 128 / 1e9}

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    n, k = X.shape
    return beta, n * np.log(rss / n) + 2 * k

def design(rs, model):
    X, y = [], []
    for r in rs:
        f = feats(r["variant"], r["ctx"])
        X.append([1, f["P"], f["S"], f["P"] * f["S"], f["L"]] if model == "P"
                 else [1, f["B_W"], f["B_KV"], f["L"]])
        y.append(r["cpu_per_token"])
    return np.array(X), np.array(y)

def fit_entry(data, vset):
    e = {}
    for model in ("P", "B"):
        X, y = design(data, model)
        _, aic = ols(X, y)
        errs = []
        for v in vset:
            tr = [r for r in data if r["variant"] != v]
            te = [r for r in data if r["variant"] == v]
            if not te: continue
            Xtr, ytr = design(tr, model); Xte, yte = design(te, model)
            b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            errs += list(np.abs(Xte @ b - yte) / yte)
        e[model] = {"aic": aic, "lovo_rel_err": float(np.mean(errs))}
    e["dAIC_P_minus_B"] = e["P"]["aic"] - e["B"]["aic"]
    e["traffic_wins"] = bool(e["B"]["lovo_rel_err"] < e["P"]["lovo_rel_err"]
                             and e["dAIC_P_minus_B"] > 10)
    e["n_reps"] = len(data)
    return e

out = {}
for label, ctxs in (("frozen_C>=4096_replication", {4096, 6144, 8192, 16384}),
                    ("sensitivity_excl_16384", {4096, 6144, 8192})):
    out[label] = {}
    for cap in (2, 4):
        data = [r for r in reps if r["cap"] == cap and r["ctx"] in ctxs
                and r["variant"] in PRIMARY]
        out[label][str(cap)] = fit_entry(data, PRIMARY)
json.dump(out, open(os.path.join(ROOT, "results/sensitivity_no16k.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
