#!/usr/bin/env python3
"""Calibration pass: FFN channel importance, attention head importance,
and per-layer Block Influence on WikiText-2 (8 seqs x 384 tokens).
Checkpoint: calibration.json (skips if present)."""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = sys.argv[1]
OUT = sys.argv[2]
if os.path.exists(OUT):
    print("calibration exists, skip"); sys.exit(0)

torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(BASE)
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(t for t in ds["text"][:400] if len(t) > 200)
ids = tok(text, return_tensors="pt")["input_ids"][0]
seqs = [ids[i * 384:(i + 1) * 384].unsqueeze(0) for i in range(8)]
print("calibration tokens:", sum(s.shape[1] for s in seqs), flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
L = model.config.num_hidden_layers
inter = model.config.intermediate_size
n_heads = model.config.num_attention_heads
n_kv = model.config.num_key_value_heads
hd = model.config.hidden_size // n_heads

ffn_imp = torch.zeros(L, inter, dtype=torch.float64)
qhead_imp = torch.zeros(L, n_heads, dtype=torch.float64)
kvhead_imp = torch.zeros(L, n_kv, dtype=torch.float64)
bi_num = torch.zeros(L, dtype=torch.float64)  # sum of (1 - cos) * tokens
bi_tok = 0

hooks = []
for li, layer in enumerate(model.model.layers):
    def mk_ffn(li):
        def h(mod, inp, out):
            a = out.detach().float()
            ffn_imp[li] += a.abs().mean(dim=tuple(range(a.dim() - 1))).double().cpu()
        return h
    hooks.append(layer.mlp.gate_proj.register_forward_hook(mk_ffn(li)))
    def mk_o(li):
        def h(mod, inp, out):
            a = inp[0].detach().float()
            a = a.reshape(*a.shape[:-1], n_heads, hd)
            qhead_imp[li] += a.abs().mean(dim=(0, 1, -1)).double().cpu()
        return h
    hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_o(li)))
    def mk_k(li):
        def h(mod, inp, out):
            a = out.detach().float()
            a = a.reshape(*a.shape[:-1], n_kv, hd)
            kvhead_imp[li] += a.abs().mean(dim=(0, 1, -1)).double().cpu()
        return h
    hooks.append(layer.self_attn.k_proj.register_forward_hook(mk_k(li)))

layer_in, layer_out = {}, {}
for li, layer in enumerate(model.model.layers):
    def mk_in(li):
        def h(mod, inp, out): layer_in[li] = inp[0].detach().float()
        return h
    def mk_out(li):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            layer_out[li] = o.detach().float()
        return h
    hooks.append(layer.register_forward_hook(mk_in(li)))
    hooks.append(layer.register_forward_hook(mk_out(li)))

with torch.inference_mode():
    for si, s in enumerate(seqs):
        _ = model(input_ids=s)
        for li in range(L):
            a, b = layer_in[li], layer_out[li]
            cos = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1)
            bi_num[li] += (1 - cos).sum().double().cpu()
        bi_tok += s.shape[1]
        print(f"seq {si} done", flush=True)

for h in hooks: h.remove()
res = {"ffn_imp": ffn_imp.tolist(), "qhead_imp": qhead_imp.tolist(),
       "kvhead_imp": kvhead_imp.tolist(),
       "block_influence": (bi_num / bi_tok).tolist(), "n_seqs": len(seqs)}
json.dump(res, open(OUT, "w"))
print("BI by layer:", [round(x, 4) for x in res["block_influence"]])
print("wrote", OUT)
