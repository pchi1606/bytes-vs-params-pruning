#!/bin/bash
# BASE model: HF reference tokens + GGUF Q4_K_M conversion + packed bytes.
# Restart-resilient: each step is guarded on its output file, so a VM
# restart only re-runs the step that was in flight.
set -u
cd ~/workspace/slm-bytes-pruning
LC=~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp

if [ ! -f models/base_hf_reference.json ]; then
./venv/bin/python - <<'PYEOF'
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
json.dump({"prompt": prompt,
           "hf_tokens": gen[0][ids.shape[1]:].tolist(),
           "hf_text": tok.decode(gen[0][ids.shape[1]:])},
          open("models/base_hf_reference.json", "w"))
print("base reference saved", flush=True)
PYEOF
else
  echo "base reference exists, skipping generation"
fi

if [ ! -f models/BASE-Q4_K_M.gguf ]; then
  if [ ! -f models/BASE-q8.gguf ]; then
    ./venv/bin/python "$LC/convert_hf_to_gguf.py" "$(cat base_path.txt)" \
      --outtype q8_0 --outfile models/BASE-q8.gguf
  fi
  "$LC/build/bin/llama-quantize" --allow-requantize models/BASE-q8.gguf \
    models/BASE-Q4_K_M.gguf Q4_K_M
  rm -f models/BASE-q8.gguf
else
  echo "BASE Q4 exists, skipping conversion"
fi

./venv/bin/python -c "
import json, os
d = json.load(open('results/packed_bytes.json'))
d['BASE'] = os.path.getsize('models/BASE-Q4_K_M.gguf')
json.dump(d, open('results/packed_bytes.json', 'w'), indent=1)
print('BASE Q4:', d['BASE'])
"
echo BASE_BUILD_DONE
