#!/usr/bin/env python3
"""Selection regret from the frozen stage-1 analysis only (no new data).

Regret(cell) = (selector mean CPU s/token - fastest primary-variant mean)
               / fastest mean * 100, per (cap, ctx), C>=4096 primary regime.
Selectors: V-mixed (traffic/byte-greedy, predeclared) and V-width
(parameter-count allocation). Fastest = min cell mean over the 6 primary
variants (V-KV-lite excluded per D4, as in the frozen primary fit).
Reads results/stage1_analysis.json; writes results/selection_regret.json.
"""
import json, statistics, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
d = json.loads((root / "results/stage1_analysis.json").read_text())
prim = set(d["primary_variants"])
rows = {}
for key, v in d["cells"].items():
    var, cap, ctx = key.rsplit("|", 2)
    if var in prim:
        rows.setdefault((int(cap), int(ctx)), {})[var] = v["cpu_tok_mean"]

out = {"definition": "regret % = (selector cell mean - fastest primary cell mean)/fastest*100; frozen cell means, C>=4096",
       "cells": {}, "per_cap_mean": {}}
for cap in (2, 4):
    rm, rw = [], []
    for ctx in (4096, 6144, 8192, 16384):
        m = rows[(cap, ctx)]
        fast = min(m, key=m.get)
        f = m[fast]
        e = {"fastest_variant": fast, "fastest_mean": f,
             "V-mixed_mean": m["V-mixed"], "V-mixed_regret_pct": (m["V-mixed"] - f) / f * 100,
             "V-width_mean": m["V-width"], "V-width_regret_pct": (m["V-width"] - f) / f * 100}
        out["cells"][f"{cap}|{ctx}"] = e
        rm.append(e["V-mixed_regret_pct"]); rw.append(e["V-width_regret_pct"])
    out["per_cap_mean"][str(cap)] = {"V-mixed_regret_pct": statistics.mean(rm),
                                     "V-mixed_max_pct": max(rm),
                                     "V-width_regret_pct": statistics.mean(rw),
                                     "V-width_max_pct": max(rw)}
(root / "results/selection_regret.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
