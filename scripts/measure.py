#!/usr/bin/env python3
"""Stage-1 decode measurement (SPEC v0.3 + Gate-2 amendment A4).
llama-server per (variant, thread-cap) session. Fixed token-ID prompts
(nested: each shorter prompt is a prefix of the longer), exactly 128
decode steps (ignore_eos, temp 0), cache capacity 20480 > 16384+128.
CPU resource time from /proc/<pid>/stat (utime+stime) around each
request; wall time secondary. Randomized paired blocks: within a
session, contexts are visited in a per-block shuffled order; every rep
is appended to results/stage1_measurements.jsonl immediately
(checkpoint/resume: cells with >= REPS records are skipped)."""
import json, os, random, signal, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.expanduser("~/workspace/publication-gap-builds/track6-synth-replication/llama.cpp/build/bin/llama-server")
PORT = 18123
# B3 runtime clause: full 7-point grid did not allow completion at the
# measured decode rate (~1.4 tok/s at 8k, cap 2) on this 2-core VM;
# 12288 dropped, keeping the B3 minimum (2048 + 6144 added).
CTXS = [512, 2048, 4096, 6144, 8192, 16384]
CAPS = [2, 4]
REPS = int(os.environ.get("REPS", "10"))
REPS_LOW = 5  # C<4096 cells are report-only (B3): 5 reps by design
def reps_for(ctx): return REPS if ctx >= 4096 else min(REPS, REPS_LOW)
GEN = 128
OUT = os.path.join(ROOT, "results", "stage1_measurements.jsonl")

def http(path, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")

def cpu_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return int(parts[11]) + int(parts[12])  # utime + stime (fields 14,15)

def rss_kb(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmHWM"): return int(line.split()[1])
    return -1

def done_counts():
    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") == "rep":
                k = (r["variant"], r["cap"], r["ctx"])
                counts[k] = counts.get(k, 0) + 1
    return counts

def start_server(variant, cap, gguf):
    proc = subprocess.Popen(
        ["taskset", "-c", "0,1", BIN, "-m", gguf, "-t", str(cap), "-tb", str(cap), "-c", "20480",
         "-np", "1", "--host", "127.0.0.1", "--port", str(PORT), "--slots"],
        stdout=open(os.path.join(ROOT, "logs", f"server-{variant}-{cap}.log"), "a"),
        stderr=subprocess.STDOUT)
    for _ in range(120):
        time.sleep(2)
        try:
            http("/health", timeout=5); return proc
        except Exception: continue
    raise RuntimeError("server did not come up")

def stop_server(proc):
    try:
        proc.send_signal(signal.SIGKILL); proc.wait(timeout=15)
    except Exception: pass
    time.sleep(2)

def main():
    variants = sys.argv[1:]
    prompts = json.load(open(os.path.join(ROOT, "results", "prompts.json")))
    counts = done_counts()
    for variant in variants:
        gguf = os.path.join(ROOT, "models", variant + "-Q4_K_M.gguf")
        for cap in CAPS:
            if all(counts.get((variant, cap, c), 0) >= reps_for(c) for c in CTXS):
                print(f"skip {variant} cap{cap} (complete)", flush=True); continue
            holder = {"proc": start_server(variant, cap, gguf)}
            primed = set()
            def completion(body, retry=True):
                # llama-server can wedge on a slot/cache transition (observed
                # 2026-10-24: 16k->2k truncation deadlocked a request). On any
                # error/timeout: restart the server, clear primed, retry once.
                try:
                    return http("/completion", body, timeout=3600)
                except Exception:
                    if not retry: raise
                    print(f"  !! request failed; restarting server ({variant} cap{cap})", flush=True)
                    stop_server(holder["proc"])
                    holder["proc"] = start_server(variant, cap, gguf)
                    primed.clear()
                    raise RuntimeError("server restarted; caller must re-prime")
            try:
                # CTX-MAJOR order (runtime adaptation, 2026-10-24): the server
                # has ONE slot; switching ctx truncates the cached tail, so
                # the original block-major shuffle re-processed up to ~12k
                # prompt tokens INSIDE later blocks' timed reps (observed:
                # a blk-1 16k rep reprocessing 12,288 tokens). That breaks
                # the frozen design's intent (timed rep = pure decode after
                # prime+warmups). So: ctxs visited in per-session shuffled
                # order, but ALL reps of a ctx run contiguously right after
                # its prime+warmups. Rep ordinal within the cell is recorded
                # as "block" (0..n), preserving the paired-block analysis.
                order = CTXS[:]
                random.Random(hash((variant, cap, "ctxmajor")) & 0xffffffff).shuffle(order)
                for ctx in order:
                    key = (variant, cap, ctx)
                    if counts.get(key, 0) >= reps_for(ctx): continue
                    ids = prompts[str(ctx)]
                    body = {"prompt": ids, "n_predict": GEN, "temperature": 0,
                            "seed": 42, "cache_prompt": True, "ignore_eos": True}
                    attempt = 0
                    while counts.get(key, 0) < reps_for(ctx):
                        attempt += 1
                        try:
                            if ctx not in primed:
                                # prime = prefill only (n_predict=1): its job is
                                # to populate the KV cache; the 128-token decode
                                # tail of the rep body pushed loaded 16k primes
                                # past the request timeout (observed 2026-10-24).
                                pbody = dict(body); pbody["n_predict"] = 1
                                c0, t0 = cpu_ticks(holder["proc"].pid), time.time()
                                r = completion(pbody)
                                rec = {"kind": "prime", "variant": variant, "cap": cap, "ctx": ctx,
                                       "cpu_s": (cpu_ticks(holder["proc"].pid) - c0) / 100.0,
                                       "wall_s": time.time() - t0, "tokens_predicted": r.get("tokens_predicted"),
                                       "prompt_n": r.get("timings", {}).get("prompt_n")}
                                with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
                                primed.add(ctx)
                                wbody = dict(body); wbody["n_predict"] = 32
                                for _ in range(5): completion(wbody)
                            c0, t0 = cpu_ticks(holder["proc"].pid), time.time()
                            r = completion(body)
                            cpu_s = (cpu_ticks(holder["proc"].pid) - c0) / 100.0
                            wall = time.time() - t0
                            blk = counts.get(key, 0)
                            rec = {"kind": "rep", "variant": variant, "cap": cap, "ctx": ctx,
                                   "block": blk, "cpu_s": cpu_s, "wall_s": wall,
                                   "cpu_per_token": cpu_s / GEN, "wall_tok_s": GEN / wall,
                                   "tokens_predicted": r.get("tokens_predicted"),
                                   "srv_pred_ms": r.get("timings", {}).get("predicted_ms"),
                                   "rss_kb_hwm": rss_kb(holder["proc"].pid)}
                            with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
                            counts[key] = counts.get(key, 0) + 1
                            attempt = 0
                            print(f"{variant} cap{cap} ctx{ctx} blk{blk}: cpu/tok={cpu_s/GEN:.4f}s walltok/s={GEN/wall:.2f} n={r.get('tokens_predicted')}", flush=True)
                        except RuntimeError:
                            if attempt >= 3:
                                print(f"  !! cell {variant} cap{cap} ctx{ctx} failed 3x; skipping (resume will retry)", flush=True)
                                break
                            continue  # server was restarted; re-prime and continue this cell
            finally:
                stop_server(holder["proc"])
    print("MEASURE DONE")

if __name__ == "__main__":
    main()
