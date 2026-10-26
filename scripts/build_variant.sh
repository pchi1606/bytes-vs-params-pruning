#!/bin/bash
# Per-variant build driver: prune -> save HF refs -> convert q8_0 ->
# delete HF -> quantize Q4_K_M -> delete q8 -> record packed bytes.
set -eo pipefail
cd ~/workspace/slm-bytes-pruning
LC=~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp
NAME=$1
echo "=== build $NAME $(date -Iseconds) ==="
./venv/bin/python scripts/prune_variants.py build "$NAME" 2>&1 | grep -v "Loading weights"
cp "models/$NAME-hf/hf_reference.json" "results/${NAME}_hf_reference.json"
cp "models/$NAME-hf/prune_summary.json" "results/${NAME}_prune_summary.json"
./venv/bin/python "$LC/convert_hf_to_gguf.py" "models/$NAME-hf" --outtype q8_0 --outfile "models/$NAME-q8.gguf" 2>&1 | tail -2
rm -rf "models/$NAME-hf"
"$LC/build/bin/llama-quantize" --allow-requantize "models/$NAME-q8.gguf" "models/$NAME-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -2
rm "models/$NAME-q8.gguf"
./venv/bin/python - "$NAME" <<'EOF'
import json, os, sys
name = sys.argv[1]
p = f"models/{name}-Q4_K_M.gguf"
d = {}
if os.path.exists("results/packed_bytes.json"): d = json.load(open("results/packed_bytes.json"))
d[name] = os.path.getsize(p)
json.dump(d, open("results/packed_bytes.json", "w"), indent=1)
print(name, "packed bytes:", d[name])
EOF
echo "=== done $NAME ==="
