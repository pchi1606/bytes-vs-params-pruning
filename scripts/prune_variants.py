#!/usr/bin/env python3
"""Stage-1 variant builder (SPEC v0.3 + Gate-2 amendment).
Modes:
  plan            -> paper accounting only; writes plans.json
  build <NAME>    -> physically prune base per plan, save HF ckpt + summary
                     + HF greedy tokens for the export check
All variants: ~25% total param cut (tied embedding counted once),
final params pairwise within 1%. Common unit space: layers / FFN
channels / KV groups (each KV group carries its GQA Q heads)."""
import json, sys, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = open(os.path.join(ROOT, "base_path.txt")).read().strip()
CAL = json.load(open(os.path.join(ROOT, "calibration.json")))
H, L0, NH, NKV, HD, INTER, VOCAB = 3072, 28, 24, 8, 128, 8192, 128256
GQA = NH // NKV
EMB = H * VOCAB
TARGET_FRAC = 0.25
BETA = 4.85 / 8.0  # Q4_K_M bytes per param (ranking only)

def layer_params(nq, nkv, inter):
    return H * HD * (2 * nq + 2 * nkv) + 3 * H * inter + 2 * H

P0 = EMB + H + L0 * layer_params(NH, NKV, INTER)
TARGET = TARGET_FRAC * P0
BI = CAL["block_influence"]
PROTECTED = {0, 1, L0 - 1}
BI_ORDER = sorted([i for i in range(L0) if i not in PROTECTED], key=lambda i: BI[i])

def solve_inter(removed_layers, keep_kv):
    """uniform keep_inter (mult of 64) minimizing |removed - TARGET|."""
    keep_layers = [i for i in range(L0) if i not in removed_layers]
    base_removed = sum(layer_params(NH, NKV, INTER) for _ in removed_layers)
    attn_removed = sum(layer_params(NH, NKV, INTER) - layer_params(keep_kv * GQA, keep_kv, INTER)
                       for _ in keep_layers)
    best = None
    for ki in range(1024, INTER + 1, 256):
        ffn_removed = sum(3 * H * (INTER - ki) for _ in keep_layers)
        tot = base_removed + attn_removed + ffn_removed
        if best is None or abs(tot - TARGET) < best[1]:
            best = (ki, abs(tot - TARGET), tot)
    return best[0], best[2]

def mkplan(removed_layers, keep_kv):
    ki, removed = solve_inter(set(removed_layers), keep_kv)
    return {"removed_layers": sorted(removed_layers), "keep_kv": keep_kv,
            "keep_inter": ki, "removed_params": removed,
            "final_params": P0 - removed, "removed_frac": removed / P0}

def greedy_mixed(S):
    """Marginal byte-greedy allocation at design context S, over UNIFORM
    units (representable in a standard Llama config / GGUF): one KV
    group in ALL remaining layers / one whole layer / a 256-ch FFN
    block in ALL remaining layers. Ratio = dBytes/dParams; bytes =
    weight bytes (BETA/param) + KV read saved at S."""
    alive = list(range(L0)); kv = NKV; inter = INTER; removed = 0.0
    def lp(k, it): return layer_params(k * GQA, k, it)
    while removed < 0.246 * P0:
        n = len(alive)
        cands = []
        if kv > 4:
            dp = n * (lp(kv, inter) - lp(kv - 1, inter)); db = dp * BETA + n * 4 * S * HD
            cands.append((db / dp, 0, ("kv",)))
        rem_layers = [i for i in alive if i not in PROTECTED]
        if rem_layers:
            li = min(rem_layers, key=lambda i: BI[i])
            dp = lp(kv, inter); db = dp * BETA + 4 * S * kv * HD
            cands.append((db / dp, 1, ("layer", li)))
        if inter > 1024:
            dp = n * 3 * H * 256; db = dp * BETA
            cands.append((db / dp, 2, ("ffn",)))
        cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        kind = cands[0][2]
        if kind[0] == "kv": removed += n * (lp(kv, inter) - lp(kv - 1, inter)); kv -= 1
        elif kind[0] == "layer": removed += lp(kv, inter); alive.remove(kind[1])
        else: removed += n * 3 * H * 256; inter -= 256
    # polish: search uniform keep_inter (mult of 256, B1 alignment) near
    # the greedy endpoint minimizing |removed - TARGET| at fixed (alive, kv)
    n = len(alive)
    def removed_at(it):
        return (L0 - n) * 0 + sum(layer_params(NH, NKV, INTER) - layer_params(kv * GQA, kv, it)
                                  for _ in alive) + sum(layer_params(NH, NKV, INTER) for _ in set(range(L0)) - set(alive))
    best = (inter, abs(removed_at(inter) - TARGET))
    for it in range(1024, INTER + 1, 256):
        e = abs(removed_at(it) - TARGET)
        if e < best[1]: best = (it, e)
    inter = best[0]; removed = removed_at(inter)
    return {"removed_layers": sorted(set(range(L0)) - set(alive)),
            "keep_kv": kv, "keep_inter": inter,
            "removed_params_greedy": removed, "removed_frac": removed / P0,
            "final_params": P0 - removed, "design_S": S,
            "uniform_kv": True, "uniform_inter": True}
    return plan

def plans():
    p = {}
    # V-width: Q-head pruning (24->16, KV untouched) is a FIXED attn
    # removal; solve FFN inter for the remainder.
    attn_removed_w = L0 * (layer_params(NH, NKV, INTER) - layer_params(16, NKV, INTER))
    best = None
    for ki in range(1024, INTER + 1, 256):
        tot = attn_removed_w + L0 * 3 * H * (INTER - ki)
        if best is None or abs(tot - TARGET) < best[1]: best = (ki, abs(tot - TARGET), tot)
    p["V-width"] = {"removed_layers": [], "keep_kv": NKV, "keep_q": 16,
                    "keep_inter": best[0], "removed_params": best[2],
                    "final_params": P0 - best[2], "removed_frac": best[2] / P0}
    # width attn removal differs (Q-only): recompute exactly below in build-verify
    p["V-depth"] = mkplan(BI_ORDER[:7], NKV)
    p["V-depth-lite"] = mkplan(BI_ORDER[:4], NKV)
    p["V-depth-alt"] = mkplan([4, 7, 10, 13, 16, 19, 22], NKV)  # B2 control: evenly-spaced, not BI
    p["V-KV"] = mkplan([], 4)
    p["V-KV-lite"] = mkplan([], 6)
    p["V-mixed"] = greedy_mixed(16384)
    p["V-mixed-shallow"] = greedy_mixed(8192)
    return p

if sys.argv[1] == "plan":
    p = plans()
    json.dump(p, open(os.path.join(ROOT, "plans.json"), "w"), indent=1)
    for k, v in p.items():
        print(k, "removed_frac=", round(v["removed_frac"], 4),
              "layers=", v.get("removed_layers"), "kv=", v.get("keep_kv"),
              "inter=", v.get("keep_inter"), "uniform=", v.get("uniform_kv"), v.get("uniform_inter"))
    finals = [v["final_params"] for v in p.values()]
    print("pairwise final-param spread:", round((max(finals) - min(finals)) / min(finals) * 100, 2), "%")
    sys.exit(0)

# ---------------- physical build ----------------
NAME = sys.argv[2]
PLAN = json.load(open(os.path.join(ROOT, "plans.json")))[NAME]
torch.set_num_threads(2)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
cfg = model.config
tok = AutoTokenizer.from_pretrained(BASE)

def ffn_prune(layer, li, keep):
    imp = torch.tensor(CAL["ffn_imp"][li])
    idx = torch.argsort(imp, descending=True)[:keep].sort().values
    mlp = layer.mlp
    with torch.no_grad():
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[idx]
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[idx]
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, idx]
    mlp.gate_proj.out_features = keep; mlp.up_proj.out_features = keep; mlp.down_proj.in_features = keep

def kv_prune(layer, li, keep_kv):
    """Keep top-keep_kv KV heads (by importance) with their full Q groups."""
    imp = torch.tensor(CAL["kvhead_imp"][li])
    keep = torch.argsort(imp, descending=True)[:keep_kv].sort().values.tolist()
    qr = torch.tensor([r for j in keep for h in range(j * GQA, (j + 1) * GQA) for r in range(h * HD, (h + 1) * HD)])
    kr = torch.tensor([r for j in keep for r in range(j * HD, (j + 1) * HD)])
    a = layer.self_attn
    with torch.no_grad():
        a.q_proj.weight.data = a.q_proj.weight.data[qr]; a.k_proj.weight.data = a.k_proj.weight.data[kr]
        a.v_proj.weight.data = a.v_proj.weight.data[kr]; a.o_proj.weight.data = a.o_proj.weight.data[:, qr]
    a.q_proj.out_features = len(qr); a.k_proj.out_features = len(kr)
    a.v_proj.out_features = len(kr); a.o_proj.in_features = len(qr)
    return keep

def q_prune(layer, li, keep_per_group=2):
    """V-width: keep top-2 Q heads per KV group; KV heads untouched."""
    imp = torch.tensor(CAL["qhead_imp"][li])
    keep_q = []
    for j in range(NKV):
        grp = list(range(j * GQA, (j + 1) * GQA))
        keep_q += sorted(grp, key=lambda h: -imp[h])[:keep_per_group]
    keep_q.sort()
    qr = torch.tensor([r for h in keep_q for r in range(h * HD, (h + 1) * HD)])
    a = layer.self_attn
    with torch.no_grad():
        a.q_proj.weight.data = a.q_proj.weight.data[qr]; a.o_proj.weight.data = a.o_proj.weight.data[:, qr]
    a.q_proj.out_features = len(qr); a.o_proj.in_features = len(qr)
    return keep_q

removed_layers = PLAN["removed_layers"]
per_layer = PLAN.get("per_layer")
keep_layers_old = [i for i in range(L0) if i not in removed_layers]
new_layers = []
for old_i in keep_layers_old:
    layer = model.model.layers[old_i]
    if per_layer:  # greedy plan: per-layer kv/inter
        st = per_layer[str(old_i)]
        if st["kv"] < NKV: kv_prune(layer, old_i, st["kv"])
        if st["inter"] < INTER: ffn_prune(layer, old_i, st["inter"])
    else:
        if NAME == "V-width":
            q_prune(layer, old_i)
        elif PLAN["keep_kv"] < NKV:
            kv_prune(layer, old_i, PLAN["keep_kv"])
        if PLAN["keep_inter"] < INTER: ffn_prune(layer, old_i, PLAN["keep_inter"])
    new_layers.append(layer)
for _ni, _layer in enumerate(new_layers):  # renumber after depth pruning (KV cache indexes by layer_idx)
    if hasattr(_layer, "layer_idx"): _layer.layer_idx = _ni
    if hasattr(_layer.self_attn, "layer_idx"): _layer.self_attn.layer_idx = _ni
model.model.layers = torch.nn.ModuleList(new_layers)
cfg.num_hidden_layers = len(new_layers)
if per_layer:
    kvs = {per_layer[str(i)]["kv"] for i in keep_layers_old}
    inters = {per_layer[str(i)]["inter"] for i in keep_layers_old}
    cfg.num_key_value_heads = kvs.pop() if len(kvs) == 1 else None
    cfg.num_attention_heads = cfg.num_key_value_heads * GQA if cfg.num_key_value_heads else None
    cfg.intermediate_size = inters.pop() if len(inters) == 1 else None
    assert cfg.num_key_value_heads and cfg.intermediate_size, "non-uniform greedy plan needs per-layer config; inspect"
else:
    cfg.num_key_value_heads = PLAN["keep_kv"]
    cfg.num_attention_heads = 16 if NAME == "V-width" else PLAN["keep_kv"] * GQA
    cfg.intermediate_size = PLAN["keep_inter"]
if NAME == "V-width": cfg.num_key_value_heads = NKV
model.config = cfg
P1 = sum(p.numel() for p in model.parameters())
print(f"{NAME}: P1={P1} removed={P0 - P1} ({100 * (P0 - P1) / P0:.2f}%)", flush=True)

# HF greedy reference tokens for the export check (fixed prompt)
prompt = "The history of the Roman Empire spans many centuries, from the founding of Rome through the republic and into the imperial period."
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.inference_mode():
    gen = model.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=128004)
ref = {"prompt": prompt, "hf_tokens": gen[0][ids.shape[1]:].tolist(),
       "hf_text": tok.decode(gen[0][ids.shape[1]:])}
OUT = os.path.join(os.environ.get("SLM_OUTROOT", os.path.join(ROOT, "models")), NAME + "-hf")
try:
    model.save_pretrained(OUT)
except Exception:
    # transformers 5.20 validator rejects head counts that do not divide
    # hidden_size even with explicit head_dim (e.g. 18 heads x 128).
    # Save under a valid count, then patch config.json to the true values.
    _rh, _rkv = cfg.num_attention_heads, cfg.num_key_value_heads
    cfg.num_attention_heads = 24
    model.save_pretrained(OUT)
    cfg.num_attention_heads = _rh
    _cj = json.load(open(os.path.join(OUT, "config.json")))
    _cj["num_attention_heads"] = _rh; _cj["num_key_value_heads"] = _rkv
    json.dump(_cj, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    print("config.json patched to true head counts", _rh, _rkv)
tok.save_pretrained(OUT)
json.dump(ref, open(os.path.join(OUT, "hf_reference.json"), "w"))
json.dump({"variant": NAME, "P0": P0, "P1": P1, "removed_frac": (P0 - P1) / P0,
           "plan": {k: v for k, v in PLAN.items() if k != "per_layer"},
           "per_layer": per_layer}, open(os.path.join(OUT, "prune_summary.json"), "w"), indent=1)
print("saved", OUT)
