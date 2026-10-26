#!/usr/bin/env python3
"""Stage-1 analysis (SPEC v0.3 + Gate-2 amendments A3/A4/B2/B3/B4
+ Amendment Round 3, 2026-10-24).
Per thread cap, on the C>=4096 subset:
  M_P: y ~ 1 + P + S + P*S + L      M_B: y ~ 1 + B_W + B_KV(S) + L
  raw: yhat = B_W + B_KV(S) as-is (relative error only)
LOVO (leave-one-variant-out) relative prediction error is the main
metric; traffic wins iff LOVO err lower AND dAIC > 10 in its favour.
Funding hurdle: paired log-ratio (V-mixed vs V-width) upper 95% CI
< 0.90 at 8k and 16k. Saturation: cap4 tok/s <= cap2 at 16k.
y = CPU (user+sys) seconds per generated token, rep-level.

FROZEN DEFINITIONS (D5, frozen 2026-10-24 BEFORE the remaining
architectures' data landed; do not change after this date):
- PRIMARY row selection: kind=rep AND "virgin" AND NOT "diagnostic".
  Pre-virgin multi-ctx-session reps are excluded. Diagnostic reps
  (D3) are summarised separately and never enter fits/cells/contrast.
- Formulas exactly as in design(): M_P [1, P, S, P*S, L];
  M_B [1, B_W, B_KV, L]; P in 1e9, S = (ctx+64)/1e3, B in 1e9 bytes.
- LOVO aggregation: mean of rep-level |pred-obs|/obs over ALL
  held-out reps (fit on the other fit-set variants' reps).
- AIC observational unit = individual rep: AIC = n ln(RSS/n) + 2k.
- Funding contrast (D1): ONLY reps carrying "pair_block", paired by
  (cap, ctx, pair_block). Fewer than 5 pairs => NOT EVALUABLE; no
  unpaired substitute. Rep-ordinal pairing is prohibited.
  [G1 amendment, Round 4 supplement 2026-10-25: pairing is by
  ATTEMPT via results/funding_attempts_manifest.json — a block's
  pair is its retained attempt's two members; restart-split and
  incomplete attempts contribute zero pairs; silent last-wins is
  prohibited. D5 fit eligibility is unaffected by D1 exclusions.]
- V-KV-lite membership of the PRIMARY fit set is governed by
  results/kvlite_logits_check.json (D4): verdict FAIL => excluded
  from primary (the excluding fit IS the sensitivity output);
  PASS => included, excluding fit reported as sensitivity."""
import hashlib, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp/gguf-py"))
import gguf

VARIANTS = ["V-width", "V-depth", "V-depth-lite", "V-depth-alt", "V-KV", "V-KV-lite", "V-mixed"]
STRUCT = {  # from plans.json / prune summaries: (layers, kv heads/layer)
    "V-width": (28, 8), "V-depth": (21, 8), "V-depth-lite": (24, 8),
    "V-depth-alt": (21, 8), "V-KV": (28, 4), "V-KV-lite": (28, 6), "V-mixed": (23, 4)}

def tensor_payload_bytes(path):
    r = gguf.GGUFReader(path)
    return int(sum(t.data.nbytes for t in r.tensors))

def tcrit(df):
    table = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
             12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
    if df in table: return table[df]
    return 1.96 if df > 30 else 2.10

def ols(X, y):
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    n, k = X.shape
    aic = n * np.log(rss / n) + 2 * k
    r2 = 1 - rss / float(np.sum((y - y.mean()) ** 2))
    return beta, pred, rss, aic, r2

def main():
    reps = []       # PRIMARY: virgin, non-diagnostic (D5 row rule)
    diag_reps = []  # D3 diagnostic remeasurement: reported separately
    primes = []
    n_excluded_previrgin = 0
    for line in open(os.path.join(ROOT, "results", "stage1_measurements.jsonl")):
        r = json.loads(line)
        # VIRGIN-PROCESS PROTOCOL (amendment 2026-10-24): only reps measured
        # in a fresh server process per cell enter the analysis. Earlier
        # multi-ctx-session reps are excluded (slot-history artifact).
        if r["kind"] == "rep":
            if not r.get("virgin"): n_excluded_previrgin += 1
            elif r.get("diagnostic"): diag_reps.append(r)
            else: reps.append(r)
        elif r["kind"] == "prime" and r.get("virgin") and not r.get("diagnostic"):
            primes.append(r)
    # D4: V-KV-lite primary-fit membership from the teacher-forced check
    kv_path = os.path.join(ROOT, "results", "kvlite_logits_check.json")
    kv_verdict = "MISSING"
    if os.path.exists(kv_path):
        kv_verdict = json.load(open(kv_path)).get("verdict", "MISSING")
    kv_in_primary = kv_verdict != "FAIL"
    primary_variants = [v for v in VARIANTS if v != "V-KV-lite" or kv_in_primary]
    sensitivity_variants = [v for v in VARIANTS if v != "V-KV-lite" or not kv_in_primary]
    packed = json.load(open(os.path.join(ROOT, "results", "packed_bytes.json")))
    summaries = {v: json.load(open(os.path.join(ROOT, "results", f"{v}_prune_summary.json"))) for v in VARIANTS}
    info = {}
    for v in VARIANTS:
        L, nkv = STRUCT[v]
        info[v] = {"L": L, "sum_kv": L * nkv, "P": summaries[v]["P1"],
                   "B_W": tensor_payload_bytes(os.path.join(ROOT, "models", f"{v}-Q4_K_M.gguf")),
                   "packed_file_bytes": packed[v]}
    def feats(v, ctx):
        S = ctx + 64  # mean occupied context over the 128-token decode (A3a)
        i = info[v]
        return {"P": i["P"] / 1e9, "S": S / 1e3, "L": i["L"],
                "B_W": i["B_W"] / 1e9, "B_KV": 4 * S * i["sum_kv"] * 128 / 1e9}
    out = {"variants": info, "cells": {}, "fits": {}, "contrast": {}, "saturation": {},
           "diagnostics": {}, "kv_lite_check_verdict": kv_verdict,
           "primary_variants": primary_variants, "sensitivity_variants": sensitivity_variants,
           "n_primary_reps": len(reps), "n_diagnostic_reps": len(diag_reps),
           "n_excluded_previrgin_reps": n_excluded_previrgin}
    # cell stats
    for r in reps:
        k = f'{r["variant"]}|{r["cap"]}|{r["ctx"]}'
        out["cells"].setdefault(k, []).append(r)
    for k, rs in out["cells"].items():
        ys = np.array([r["cpu_per_token"] for r in rs]); ws = np.array([r["wall_tok_s"] for r in rs])
        m, sd = ys.mean(), ys.std(ddof=1) if len(ys) > 1 else 0.0
        ci = tcrit(len(ys) - 1) * sd / np.sqrt(len(ys)) if len(ys) > 1 else float("nan")
        out["cells"][k] = {"n": len(rs), "cpu_tok_mean": float(m), "cpu_tok_sd": float(sd),
                           "cpu_tok_ci95": float(ci), "wall_tok_s_mean": float(ws.mean()),
                           "rss_kb_hwm_max": max(r["rss_kb_hwm"] for r in rs)}
    # saturation at 16k: cap4 wall tok/s vs cap2
    for v in VARIANTS:
        a = out["cells"].get(f"{v}|2|16384", {}).get("wall_tok_s_mean")
        b = out["cells"].get(f"{v}|4|16384", {}).get("wall_tok_s_mean")
        if a and b: out["saturation"][v] = {"tok_s_cap2": a, "tok_s_cap4": b, "saturated": bool(b <= a)}
    # fits + LOVO per cap, C>=4096 (definitions FROZEN, see header/D5)
    def design(rs, model):
        X, y = [], []
        for r in rs:
            f = feats(r["variant"], r["ctx"])
            X.append([1, f["P"], f["S"], f["P"] * f["S"], f["L"]] if model == "P"
                     else [1, f["B_W"], f["B_KV"], f["L"]])
            y.append(r["cpu_per_token"])
        return np.array(X), np.array(y)
    def fit_entry(data, vset):
        entry = {}
        for model in ("P", "B"):
            X, y = design(data, model)
            beta, pred, rss, aic, r2 = ols(X, y)
            mae = float(np.mean(np.abs(y - pred)))
            # LOVO: mean rep-level relative error over all held-out reps
            errs = []
            for v in vset:
                tr = [r for r in data if r["variant"] != v]; te = [r for r in data if r["variant"] == v]
                if not te: continue
                Xtr, ytr = design(tr, model); Xte, yte = design(te, model)
                b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
                errs += list(np.abs(Xte @ b - yte) / yte)
            entry[model] = {"beta": [float(x) for x in beta], "r2": r2, "aic": aic,
                            "mae_insample": mae, "lovo_rel_err": float(np.mean(errs)),
                            "L_coef": float(beta[-1])}
        # raw single-cost predictor as-is, reported separately (D5)
        errs = []
        for r in data:
            f = feats(r["variant"], r["ctx"])
            errs.append(abs((f["B_W"] + f["B_KV"]) - r["cpu_per_token"]) / r["cpu_per_token"])
        entry["raw_B_rel_err_UNSCALED"] = float(np.mean(errs))  # GB vs s/token: scale-free check only via ranking
        entry["dAIC_P_minus_B"] = entry["P"]["aic"] - entry["B"]["aic"]
        entry["traffic_wins"] = bool(entry["B"]["lovo_rel_err"] < entry["P"]["lovo_rel_err"]
                                     and entry["dAIC_P_minus_B"] > 10)
        entry["n_reps"] = len(data)
        return entry
    for cap in (2, 4):
        all_data = [r for r in reps if r["cap"] == cap and r["ctx"] >= 4096]
        pdata = [r for r in all_data if r["variant"] in primary_variants]
        sdata = [r for r in all_data if r["variant"] in sensitivity_variants]
        entry = fit_entry(pdata, primary_variants)
        entry["sensitivity"] = fit_entry(sdata, sensitivity_variants)
        out["fits"][str(cap)] = entry
    # paired contrast V-mixed vs V-width (funding hurdle), per cap and
    # ctx in {8192, 16384}. D1 pairing is MANIFEST-DRIVEN (G1, Round 4
    # supplement): pairs come only from the retained attempt of each
    # block in results/funding_attempts_manifest.json (build via
    # scripts/build_manifest.py). Silent last-wins selection is
    # prohibited; the md5 gate refuses a stale manifest. <5 pairs =>
    # NOT EVALUABLE (no substitute). Analysis code version:
    # G1-amended 2026-10-25 (supersedes the F1 version, preserved as
    # scripts/analyze_f1_superseded.py; prior output preserved as
    # results/stage1_analysis_STALE-2026-10-25-0210.json).
    _jsonl_path = os.path.join(ROOT, "results", "stage1_measurements.jsonl")
    _md5 = hashlib.md5(open(_jsonl_path, "rb").read()).hexdigest()
    manifest = json.load(open(os.path.join(ROOT, "results", "funding_attempts_manifest.json")))
    if manifest["jsonl_md5_at_build"] != _md5:
        raise SystemExit("funding manifest is stale (JSONL md5 mismatch): "
                         "regenerate with scripts/build_manifest.py before analysing")
    out["funding_attempts"] = {"manifest_version": manifest["version"],
                                "accounting": manifest["attempt_accounting"]}
    def manifest_pairs(cap, ctx):
        a, b, repaired_blocks = {}, {}, []
        for bl in manifest["blocks"]:
            if bl["cap"] == cap and bl["ctx"] == ctx and bl["pair"]:
                a[bl["pair_block"]] = bl["pair"]["V-mixed"]
                b[bl["pair_block"]] = bl["pair"]["V-width"]
                att = next(x for x in bl["attempts"]
                           if x["attempt_id"] == bl["retained_attempt"])
                if att["status"] == "repaired_valid":
                    repaired_blocks.append(bl["pair_block"])
        return a, b, repaired_blocks
    for cap in (2, 4):
        for ctx in (8192, 16384):
            a, b, repaired = manifest_pairs(cap, ctx)
            common = sorted(set(a) & set(b))
            f = feats("V-mixed", ctx); g = feats("V-width", ctx)
            pred_ratio = (f["B_W"] + f["B_KV"]) / (g["B_W"] + g["B_KV"])
            sat = bool(out["saturation"].get("V-width", {}).get("saturated") if ctx == 16384 else False)
            if len(common) >= 5:
                lr = np.log([a[k] / b[k] for k in common])
                m, sd = lr.mean(), lr.std(ddof=1)
                half = tcrit(len(common) - 1) * sd / np.sqrt(len(common))
                out["contrast"][f"{cap}|{ctx}"] = {
                    "evaluable": True, "n_pairs": len(common), "ratio_mean": float(np.exp(m)),
                    "ratio_ci95": [float(np.exp(m - half)), float(np.exp(m + half))],
                    "upper_lt_0.90": bool(np.exp(m + half) < 0.90),
                    "repaired_blocks": repaired,
                    "traffic_predicted_ratio": float(pred_ratio),
                    "saturation_regime": sat}
            else:
                out["contrast"][f"{cap}|{ctx}"] = {
                    "evaluable": False, "n_pairs": len(common),
                    "repaired_blocks": repaired,
                    "reason": "fewer than 5 genuine pair_blocks (D1); funding hurdle "
                              "NOT VALIDLY EVALUABLE at this (cap, ctx); no unpaired "
                              "substitute computed",
                    "traffic_predicted_ratio": float(pred_ratio),
                    "saturation_regime": sat}
    # D3 diagnostics: summarised separately against the ORIGINAL cell means
    diag_cells = {}
    for r in diag_reps:
        k = f'{r["variant"]}|{r["cap"]}|{r["ctx"]}'
        diag_cells.setdefault(k, []).append(r["cpu_per_token"])
    for k, ys_list in diag_cells.items():
        ys = np.array(ys_list)
        m, sd = ys.mean(), ys.std(ddof=1) if len(ys) > 1 else 0.0
        ci = tcrit(len(ys) - 1) * sd / np.sqrt(len(ys)) if len(ys) > 1 else float("nan")
        orig = out["cells"].get(k, {}).get("cpu_tok_mean")
        out["diagnostics"][k] = {"n": len(ys), "cpu_tok_mean": float(m), "cpu_tok_sd": float(sd),
                                 "cpu_tok_ci95": float(ci), "original_cell_mean": orig,
                                 "ratio_diag_over_original": float(m / orig) if orig else None}
    # prefill (prime) summary
    pf = {}
    for r in primes:
        pf.setdefault(f'{r["variant"]}|{r["cap"]}|{r["ctx"]}', []).append(r["cpu_s"])
    out["prefill_cpu_s_mean"] = {k: float(np.mean(v)) for k, v in pf.items()}
    json.dump(out, open(os.path.join(ROOT, "results", "stage1_analysis.json"), "w"), indent=1)
    print(json.dumps({"fits": out["fits"], "contrast": out["contrast"],
                      "saturation": out["saturation"], "diagnostics": out["diagnostics"],
                      "kv_lite_check_verdict": kv_verdict,
                      "primary_variants": primary_variants}, indent=1))

if __name__ == "__main__":
    main()
