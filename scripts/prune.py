#!/usr/bin/env python3
"""Structured pruning. Usage: prune.py BASE CALIB OUTDIR ARM [TARGET_REMOVAL]
ARM=A: uniform width pruning, cuts split FFN/attention proportional to param share, no depth cut.
ARM=B: depth-first (lowest Block-Influence layers), remainder FFN width pruning.
TARGET_REMOVAL: exact param-removal target (used to match arms); default 25% of total."""
import json, sys, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, CALIB, OUT, ARM = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(2)
cal = json.load(open(CALIB))
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
cfg = model.config
L0, H, INTER = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
NH, NKV, HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
GQA = NH // NKV

def nparams(m): return sum(p.numel() for p in m.parameters())
P0 = nparams(model)
TARGET = int(sys.argv[5]) if len(sys.argv) > 5 else int(round(0.25 * P0))
print(f"P0={P0} target_removal={TARGET}", flush=True)

ffn_layer = 3 * H * INTER
attn_layer = H * (NH * HD) + 2 * H * (NKV * HD) + (NH * HD) * H
FFN_TOT, ATTN_TOT = ffn_layer * L0, attn_layer * L0

def keep_channels(li, keep):
    imp = torch.tensor(cal["ffn_imp"][li])
    return torch.argsort(imp, descending=True)[:keep].sort().values

def apply_ffn_prune(layer, keep_idx):
    mlp = layer.mlp
    k = len(keep_idx)
    with torch.no_grad():
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[keep_idx]
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[keep_idx]
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, keep_idx]
    mlp.gate_proj.out_features = k; mlp.up_proj.out_features = k; mlp.down_proj.in_features = k

def apply_head_prune(layer, li, keep_kv):
    attn = layer.self_attn
    kv_imp = torch.tensor(cal["kvhead_imp"][li])
    keep_kv_idx = torch.argsort(kv_imp, descending=True)[:keep_kv].sort().values.tolist()
    keep_q = [h for j in keep_kv_idx for h in range(j * GQA, (j + 1) * GQA)]
    qr = torch.tensor([r for h in keep_q for r in range(h * HD, (h + 1) * HD)])
    kr = torch.tensor([r for j in keep_kv_idx for r in range(j * HD, (j + 1) * HD)])
    with torch.no_grad():
        attn.q_proj.weight.data = attn.q_proj.weight.data[qr]
        if attn.q_proj.bias is not None: attn.q_proj.bias.data = attn.q_proj.bias.data[qr]
        attn.k_proj.weight.data = attn.k_proj.weight.data[kr]
        if attn.k_proj.bias is not None: attn.k_proj.bias.data = attn.k_proj.bias.data[kr]
        attn.v_proj.weight.data = attn.v_proj.weight.data[kr]
        if attn.v_proj.bias is not None: attn.v_proj.bias.data = attn.v_proj.bias.data[kr]
        attn.o_proj.weight.data = attn.o_proj.weight.data[:, qr]
    attn.q_proj.out_features = len(qr); attn.k_proj.out_features = len(kr)
    attn.v_proj.out_features = len(kr); attn.o_proj.in_features = len(qr)
    return keep_kv_idx

removed_layers, keep_kv_final = [], None
if ARM == "A":
    ffn_cut = TARGET * FFN_TOT // (FFN_TOT + ATTN_TOT)
    attn_cut = TARGET - ffn_cut
    # choose kv-head keep minimizing |attn removal - attn_cut|; attn removal per layer at keep k:
    best = None
    for k in range(NKV, 2, -1):
        rem = (attn_layer - (H * (k * GQA * HD) + 2 * H * (k * HD) + (k * GQA * HD) * H)) * L0
        if best is None or abs(rem - attn_cut) < best[1]: best = (k, abs(rem - attn_cut), rem)
    keep_kv_final, _, attn_removed = best
    ffn_cut = TARGET - attn_removed
    per_layer_ffn_cut = ffn_cut // L0
    keep_inter = INTER - per_layer_ffn_cut // (3 * H)
    keep_inter = max(512, int(round(keep_inter / 128)) * 128)
    for li, layer in enumerate(model.model.layers):
        apply_ffn_prune(layer, keep_channels(li, keep_inter))
        apply_head_prune(layer, li, keep_kv_final)
    cfg.intermediate_size = keep_inter
    cfg.num_attention_heads = keep_kv_final * GQA
    cfg.num_key_value_heads = keep_kv_final
elif ARM == "B":
    bi = cal["block_influence"]
    cand = sorted(range(2, L0 - 1), key=lambda i: bi[i])  # protect first 2 + last layer
    depth_removed, n_rm = 0, 0
    for li in cand:  # majority of cut from depth: stop once >= 55% of target
        if depth_removed >= 0.55 * TARGET: break
        depth_removed += ffn_layer + attn_layer + 2 * H  # + 2 layernorms
        n_rm += 1
    removed_layers = sorted(cand[:n_rm])
    keep_layers = [l for i, l in enumerate(model.model.layers) if i not in removed_layers]
    model.model.layers = torch.nn.ModuleList(keep_layers)
    cfg.num_hidden_layers = len(keep_layers)
    rem_needed = TARGET - depth_removed
    ffn_rem = sum(3 * H * INTER for _ in keep_layers)
    per_layer_cut = rem_needed // len(keep_layers)
    keep_inter = INTER - per_layer_cut // (3 * H)
    keep_inter = max(512, int(round(keep_inter / 128)) * 128)
    for new_li, layer in enumerate(keep_layers):
        old_li = [i for i in range(L0) if i not in removed_layers][new_li]
        apply_ffn_prune(layer, keep_channels(old_li, keep_inter))
    cfg.intermediate_size = keep_inter
else:
    raise ValueError(ARM)

P1 = nparams(model)
print(f"P1={P1} removed={P0 - P1} ({100 * (P0 - P1) / P0:.2f}%) removed_layers={removed_layers} keep_kv={keep_kv_final} keep_inter={cfg.intermediate_size}", flush=True)
model.config = cfg
model.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
json.dump({"arm": ARM, "P0": P0, "P1": P1, "removed": P0 - P1,
           "removed_layers": removed_layers, "keep_kv_heads": keep_kv_final,
           "intermediate_size": cfg.intermediate_size, "num_layers": cfg.num_hidden_layers},
          open(OUT + "/prune_summary.json", "w"), indent=1)
print("saved", OUT)
