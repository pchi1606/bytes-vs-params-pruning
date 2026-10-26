#!/usr/bin/env python3
"""V-KV-lite teacher-forced HF-vs-GGUF logits check (SPEC D4, 2026-10-24).
Identical fixed token sequences for BASE and V-KV-lite. Per evaluated
position: teacher-forced next-token distribution from HF (full logits)
vs llama-server GGUF (n_probs top-10). Metrics: mean NLL of the true
next token under each side, dNLL = NLL_GGUF - NLL_HF, truncated KL
(HF || GGUF) over the union of top-10 supports, top-1 agreement.
PREDECLARED verdict: PASS iff dNLL(KVLITE) <= dNLL(BASE) + 0.05 AND
meanKL(KVLITE) <= 2 * meanKL(BASE) AND top1(KVLITE) >= top1(BASE) - 0.10.
Writes results/kvlite_logits_check.json. HF phase first (dumps
per-position distributions), then the GGUF server phase, per model,
so peak RAM stays bounded."""
import gc, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import ROOT, start_server, stop_server, http

SEQ_LEN = 65
POS = list(range(16, 64))  # predict token at i from prefix [:i]
MODELS = ["BASE", "V-KV-lite"]

def sequences():
    prompts = json.load(open(os.path.join(ROOT, "results", "prompts.json")))
    ids = prompts["512"]
    return [ids[k * SEQ_LEN:(k + 1) * SEQ_LEN] for k in range(3)]

def hf_phase(name, path, seqs):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.eval()
    return teacher_force(model, seqs)

def hf_phase_inmemory(seqs):
    """V-KV-lite's disk checkpoint cannot be re-loaded: transformers
    5.20.2 validates 3072 % 18 != 0 even with explicit head_dim (the
    same wall the build hit). Build the HF side by the ORIGINAL
    in-memory route: run scripts/prune_variants.py (deterministic;
    P1 and reference tokens verified identical to the shipped build)
    and teacher-force its in-memory model."""
    import runpy
    argv0 = sys.argv
    sys.argv = ["prune_variants.py", "build", "V-KV-lite"]
    try:
        ns = runpy.run_path(os.path.join(ROOT, "scripts", "prune_variants.py"))
    finally:
        sys.argv = argv0
    return teacher_force(ns["model"], seqs)

def teacher_force(model, seqs):
    import torch
    model.eval()
    out = {}
    with torch.inference_mode():
        for si, seq in enumerate(seqs):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x).logits[0].float()  # [L, V]
            logp = torch.log_softmax(logits, dim=-1)
            for i in POS:
                lp = logp[i - 1]
                top = torch.topk(lp, 10)
                out[f"{si}:{i}"] = {
                    "true": seq[i], "nll": float(-lp[seq[i]]),
                    "top1": int(top.indices[0]),
                    "top_ids": [int(t) for t in top.indices],
                    "top_logp": [float(t) for t in top.values]}
    del model
    gc.collect()
    return out

def gguf_phase(name, gguf, seqs):
    proc = start_server(name, 2, gguf)
    out = {}
    try:
        for si, seq in enumerate(seqs):
            for i in POS:
                r = http("/completion", {"prompt": seq[:i], "n_predict": 1,
                                         "temperature": 0, "n_probs": 10,
                                         "cache_prompt": True}, timeout=3600)
                probs = r["completion_probabilities"][0]["top_logprobs"]
                top_ids, top_p = [], []
                for e in probs:
                    tid = e.get("id", e.get("tok_id"))
                    if tid is None:
                        raise RuntimeError(f"n_probs entry lacks token id: {e}")
                    top_ids.append(int(tid)); top_p.append(float(np.exp(e["logprob"])))
                out[f"{si}:{i}"] = {"true": seq[i], "top1": top_ids[0],
                                    "top_ids": top_ids, "top_p": top_p}
    finally:
        stop_server(proc)
    return out

def compare(hf, gg):
    dnll, kls, agree, n = [], [], 0, 0
    nll_hf, nll_gg = [], []
    for k, h in hf.items():
        g = gg.get(k)
        if g is None: continue
        n += 1
        nll_hf.append(h["nll"])
        # GGUF NLL of the true token: exact if in top-10, else lower-bounded
        if h["true"] in g["top_ids"]:
            j = g["top_ids"].index(h["true"])
            nll_gg.append(-np.log(max(g["top_p"][j], 1e-12)))
            dnll.append(nll_gg[-1] - h["nll"])
        # truncated KL(HF||GGUF) over union of top-10 supports
        hp = {t: float(np.exp(l)) for t, l in zip(h["top_ids"], h["top_logp"])}
        gp = {t: p for t, p in zip(g["top_ids"], g["top_p"])}
        rest_h = max(1.0 - sum(hp.values()), 0.0); rest_g = max(1.0 - sum(gp.values()), 0.0)
        kl = 0.0
        for t in set(hp) | set(gp):
            ph, pg = hp.get(t, 0.0), gp.get(t, 0.0)
            if ph > 0: kl += ph * np.log(ph / max(pg, 1e-12))
        if rest_h > 0: kl += rest_h * np.log(rest_h / max(rest_g, 1e-12))
        kls.append(kl)
        agree += int(h["top1"] == g["top1"])
    return {"n_positions": n, "nll_hf_mean": float(np.mean(nll_hf)),
            "nll_gguf_mean_exact_subset": float(np.mean(nll_gg)) if nll_gg else None,
            "n_true_in_gguf_top10": len(dnll),
            "dnll_mean": float(np.mean(dnll)) if dnll else None,
            "kl_mean": float(np.mean(kls)), "top1_agreement": agree / n}

def main():
    seqs = sequences()
    base_path = open(os.path.join(ROOT, "base_path.txt")).read().strip()
    paths = {"BASE": base_path, "V-KV-lite": os.path.join(ROOT, "models", "V-KV-lite-hf")}
    ggufs = {"BASE": os.path.join(ROOT, "models", "BASE-Q4_K_M.gguf"),
             "V-KV-lite": os.path.join(ROOT, "models", "V-KV-lite-Q4_K_M.gguf")}
    res = {"sequences": "results/prompts.json ['512'] 3x65-token slices",
           "positions": [POS[0], POS[-1]], "models": {}}
    for name in MODELS:
        print(f"HF phase {name} ...", flush=True)
        hf = hf_phase_inmemory(seqs) if name == "V-KV-lite" else hf_phase(name, paths[name], seqs)
        json.dump(hf, open(os.path.join(ROOT, "results", f"kvlite_check_hf_{name}.json"), "w"))
        print(f"GGUF phase {name} ...", flush=True)
        gg = gguf_phase(name, ggufs[name], seqs)
        res["models"][name] = compare(hf, gg)
        print(name, json.dumps(res["models"][name]), flush=True)
    b, k = res["models"]["BASE"], res["models"]["V-KV-lite"]
    res["rule"] = "PASS iff dnll(KVLITE) <= dnll(BASE)+0.05 AND kl(KVLITE) <= 2*kl(BASE) AND top1(KVLITE) >= top1(BASE)-0.10"
    ok = (k["dnll_mean"] is not None and b["dnll_mean"] is not None
          and k["dnll_mean"] <= b["dnll_mean"] + 0.05
          and k["kl_mean"] <= 2 * b["kl_mean"]
          and k["top1_agreement"] >= b["top1_agreement"] - 0.10)
    res["verdict"] = "PASS" if ok else "FAIL"
    json.dump(res, open(os.path.join(ROOT, "results", "kvlite_logits_check.json"), "w"), indent=1)
    print("VERDICT:", res["verdict"])

if __name__ == "__main__":
    main()
