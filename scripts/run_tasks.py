#!/usr/bin/env python3
"""Run the 40-task suite against one GGUF with llama-cli at a fixed thread cap.
Captures per-task user CPU seconds + peak RSS via /usr/bin/time -v.
Outputs raw completions to OUTDIR/<id>.txt and a timings.json."""
import json, subprocess, sys, os, re

LLAMA = os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp/build/bin/llama-cli")
model, threads, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
tasks = json.load(open(os.path.join(os.path.dirname(__file__), "tasks.committed.json")))
timings = {}
for t in tasks:
    cmd = ["/usr/bin/time", "-v", LLAMA, "-m", model, "-t", str(threads), "-c", "4096",
           "-n", str(t["max_tokens"]), "-p", t["prompt"], "--temp", "0", "-s", "42",
           "-no-cnv", "--no-display-prompt", "--simple-io"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = r.stdout
    # llama-cli echoes the prompt once despite --no-display-prompt in some builds; strip it
    if out.startswith(t["prompt"]):
        out = out[len(t["prompt"]):]
    open(os.path.join(outdir, t["id"] + ".txt"), "w").write(out)
    m_user = re.search(r"User time \(seconds\): ([\d.]+)", r.stderr)
    m_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
    timings[t["id"]] = {"user_s": float(m_user.group(1)) if m_user else None,
                        "max_rss_kb": int(m_rss.group(1)) if m_rss else None}
    print(t["id"], timings[t["id"]], flush=True)
json.dump(timings, open(os.path.join(outdir, "timings.json"), "w"), indent=1)
tot = sum(v["user_s"] for v in timings.values() if v["user_s"])
print(f"TOTAL user CPU {tot:.1f}s over {len(timings)} tasks -> {tot/len(timings):.2f} CPU-s/task")
