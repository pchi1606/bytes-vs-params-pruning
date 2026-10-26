#!/usr/bin/env python3
"""Decode tok/s probe: fixed prompt, 128 generated tokens, parse llama-cli timing lines."""
import subprocess, sys, re, os

LLAMA = os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp/build/bin/llama-cli")
model, threads = sys.argv[1], sys.argv[2]
prompt = ("The history of the Roman Empire spans many centuries, from the founding of Rome "
          "through the republic and into the imperial period, shaping law, engineering, "
          "language and trade across Europe and the Mediterranean world.")
cmd = [LLAMA, "-m", model, "-t", threads, "-c", "4096", "-n", "128", "-p", prompt,
       "--temp", "0", "-s", "42", "-no-cnv", "--no-display-prompt", "--simple-io"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
m = re.findall(r"eval time =.*?([\d.]+) tokens per second", r.stderr)
print("decode_tok_s:", m[-1] if m else "PARSE_FAIL")
if not m: print(r.stderr[-1500:])
