#!/bin/bash
# Variant build staging the HF checkpoint on the root overlay (/root/slm-tmp,
# ephemeral) because the workspace volume is nearly full. GGUFs land in workspace.
set -eo pipefail
cd ~/workspace/slm-bytes-pruning
LC=~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp
NAME=$1
TMP=/root/slm-tmp
echo "=== build $NAME (tmp-staged) $(date -Iseconds) ==="
SLM_OUTROOT=$TMP ./venv/bin/python scripts/prune_variants.py build "$NAME" 2>&1 | grep -v "Loading weights"
cp "$TMP/$NAME-hf/hf_reference.json" "results/${NAME}_hf_reference.json"
cp "$TMP/$NAME-hf/prune_summary.json" "results/${NAME}_prune_summary.json"
./venv/bin/python "$LC/convert_hf_to_gguf.py" "$TMP/$NAME-hf" --outtype q8_0 --outfile "$TMP/$NAME-q8.gguf" 2>&1 | tail -1
rm -rf "$TMP/$NAME-hf"
"$LC/build/bin/llama-quantize" --allow-requantize "$TMP/$NAME-q8.gguf" "models/$NAME-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -1
rm "$TMP/$NAME-q8.gguf"
./venv/bin/python - "$NAME" <<'PYEOF'
import json, os, sys
name = sys.argv[1]
d = json.load(open("results/packed_bytes.json")) if os.path.exists("results/packed_bytes.json") else {}
d[name] = os.path.getsize(f"models/{name}-Q4_K_M.gguf")
json.dump(d, open("results/packed_bytes.json", "w"), indent=1)
print(name, "packed bytes:", d[name])
PYEOF
echo "=== done $NAME ==="
