#!/usr/bin/env python3
"""Follow-up to check_kvlite_logits.py: the combined run was OOM-killed
during the V-KV-lite GGUF phase (HF dumps for both models survived in
results/kvlite_check_hf_<model>.json). This process holds NO HF model:
it loads the dumps, runs both GGUF phases, compares, and writes
results/kvlite_logits_check.json with the same predeclared D4 rule."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_kvlite_logits import gguf_phase, compare, sequences, MODELS
from measure import ROOT

def main():
    seqs = sequences()
    ggufs = {"BASE": os.path.join(ROOT, "models", "BASE-Q4_K_M.gguf"),
             "V-KV-lite": os.path.join(ROOT, "models", "V-KV-lite-Q4_K_M.gguf")}
    res = {"sequences": "results/prompts.json ['512'] 3x65-token slices",
           "positions": [16, 63], "models": {}}
    for name in MODELS:
        hf = json.load(open(os.path.join(ROOT, "results", f"kvlite_check_hf_{name}.json")))
        hf = {k: v for k, v in hf.items()}
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
