#!/bin/bash
# Sequential stage-1 builds, checkpointed by GGUF existence.
cd ~/workspace/slm-bytes-pruning
LC=~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp
for NAME in V-width V-depth V-depth-lite V-depth-alt V-KV V-KV-lite V-mixed; do
  if [ -f "models/$NAME-Q4_K_M.gguf" ]; then echo "skip $NAME (exists)"; continue; fi
  bash scripts/build_variant.sh "$NAME" || echo "BUILD FAILED: $NAME"
done
# Base GGUF + HF reference (export-check reference level)
if [ ! -f models/BASE-Q4_K_M.gguf ]; then
  ./venv/bin/python - <<'EOF'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE = open("base_path.txt").read().strip()
torch.set_num_threads(2)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(BASE)
prompt = "The history of the Roman Empire spans many centuries, from the founding of Rome through the republic and into the imperial period."
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.inference_mode():
    gen = m.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=128004)
json.dump({"prompt": prompt, "hf_tokens": gen[0][ids.shape[1]:].tolist(),
           "hf_text": tok.decode(gen[0][ids.shape[1]:])}, open("models/base_hf_reference.json", "w"))
print("base reference saved")
EOF
  ./venv/bin/python "$LC/convert_hf_to_gguf.py" "$(cat base_path.txt)" --outtype q8_0 --outfile models/BASE-q8.gguf 2>&1 | tail -1
  "$LC/build/bin/llama-quantize" models/BASE-q8.gguf models/BASE-Q4_K_M.gguf Q4_K_M 2>&1 | tail -1
  rm models/BASE-q8.gguf
  ./venv/bin/python -c "
import json, os
d = json.load(open('results/packed_bytes.json')) if os.path.exists('results/packed_bytes.json') else {}
d['BASE'] = os.path.getsize('models/BASE-Q4_K_M.gguf'); json.dump(d, open('results/packed_bytes.json','w'), indent=1)
print('BASE packed bytes:', d['BASE'])"
fi
echo "ALL BUILDS DONE"
df -h . | tail -1
