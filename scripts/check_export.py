#!/usr/bin/env python3
"""HF-vs-GGUF export check (amendment A4): for each variant, greedy-
generate 32 tokens from the fixed prompt with llama-cli (temp 0) and
compare token IDs against the HF reference saved at build time
(hf_reference.json). Re-tokenization of CLI text is approximate at
boundaries; agreement is reported alongside the base model's own
HF-vs-GGUF agreement as the reference level. Coherence is eyeballed
from the printed text in the log."""
import json, os, subprocess, sys
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp/build/bin/llama-cli")
BASE = open(os.path.join(ROOT, "base_path.txt")).read().strip()
tok = AutoTokenizer.from_pretrained(BASE)

def cli_gen(gguf, prompt):
    cmd = [CLI, "-m", gguf, "-t", "2", "-c", "2048", "-n", "32", "-p", prompt,
           "--temp", "0", "-s", "42", "--single-turn", "--no-display-prompt", "--simple-io"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = r.stdout
    marker = "> " + prompt
    i = out.find(marker)
    if i >= 0:
        out = out[i + len(marker):]
    else:
        j = out.find(prompt)
        if j >= 0: out = out[j + len(prompt):]
    for cut in ("\n[ Prompt:", "\n\nExiting", "[ Prompt:"):
        k = out.find(cut)
        if k >= 0: out = out[:k]
    return out.strip()

def agree(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / max(n, 1), n

out = {}
_prev = os.path.join(ROOT, "results", "export_check.json")
if os.path.exists(_prev):
    try: out = json.load(open(_prev))
    except Exception: out = {}
for name in sys.argv[1:]:
    if name == "BASE":
        ref = json.load(open(os.path.join(ROOT, "models", "base_hf_reference.json")))
        gguf = os.path.join(ROOT, "models", "BASE-Q4_K_M.gguf")
    else:
        ref = json.load(open(os.path.join(ROOT, "models", name + "-hf", "hf_reference.json"))) \
            if os.path.exists(os.path.join(ROOT, "models", name + "-hf", "hf_reference.json")) \
            else json.load(open(os.path.join(ROOT, "results", name + "_hf_reference.json")))
        gguf = os.path.join(ROOT, "models", name + "-Q4_K_M.gguf")
    text = cli_gen(gguf, ref["prompt"])
    gen_text = text[len(ref["prompt"]):] if text.startswith(ref["prompt"]) else text
    gguf_tokens = tok(gen_text, add_special_tokens=False)["input_ids"][:32]
    ag, n = agree(ref["hf_tokens"], gguf_tokens)
    out[name] = {"agreement": ag, "n_compared": n, "gguf_text": gen_text[:200]}
    print(f"{name}: agreement={ag:.2f} over {n} tokens | text: {gen_text[:120]!r}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "export_check.json"), "w"), indent=1)
print("wrote results/export_check.json")
