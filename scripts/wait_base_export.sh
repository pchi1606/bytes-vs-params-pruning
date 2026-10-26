#!/bin/bash
# Wait for BASE_BUILD_DONE, then run the full export check (7 variants + BASE).
cd ~/workspace/slm-bytes-pruning
for i in $(seq 1 240); do
  grep -q BASE_BUILD_DONE logs/base-build.log 2>/dev/null && break
  sleep 30
done
if grep -q BASE_BUILD_DONE logs/base-build.log 2>/dev/null; then
  ./venv/bin/python scripts/check_export.py V-width V-depth V-depth-lite V-depth-alt V-KV V-KV-lite V-mixed BASE > logs/export-check3.log 2>&1
  echo EXPORT_CHECK_DONE
else
  echo BASE_BUILD_WAIT_TIMEOUT
fi
