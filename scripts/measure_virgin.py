#!/usr/bin/env python3
"""Stage-1 measurement, VIRGIN-PROCESS protocol (amendment 2026-10-24).
One fresh llama-server process per (variant, cap, ctx): prime (prefill,
n_predict=1) -> 5 warmups (32 tok) -> REPS reps (exactly 128 decode steps,
fixed token IDs, cache_prompt). Records carry "virgin": true; analyze.py
uses virgin reps ONLY (multi-ctx-session reps are excluded: decode speed
at a ctx proved to depend on the slot's ctx-switch history — V-width
cap2 16k measured 1.62-1.71 CPU-s/tok in switched sessions vs 0.62-0.64
in a near-virgin session, server-reported decode times agreeing).
Checkpoint/resume via results/stage1_measurements.jsonl."""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import (ROOT, OUT, CTXS, CAPS, GEN, start_server, stop_server,
                     http, cpu_ticks, rss_kb)

REPS = int(os.environ.get("REPS", "5"))

def virgin_counts():
    counts = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") == "rep" and r.get("virgin"):
                k = (r["variant"], r["cap"], r["ctx"])
                counts[k] = counts.get(k, 0) + 1
    return counts

def main():
    variants = sys.argv[1:]
    prompts = json.load(open(os.path.join(ROOT, "results", "prompts.json")))
    counts = virgin_counts()
    for variant in variants:
        gguf = os.path.join(ROOT, "models", variant + "-Q4_K_M.gguf")
        for cap in CAPS:
            order = CTXS[:]
            random.Random(hash((variant, cap, "virgin")) & 0xffffffff).shuffle(order)
            for ctx in order:
                key = (variant, cap, ctx)
                if counts.get(key, 0) >= REPS:
                    print(f"skip {variant} cap{cap} ctx{ctx} (virgin complete)", flush=True); continue
                ids = prompts[str(ctx)]
                body = {"prompt": ids, "n_predict": GEN, "temperature": 0,
                        "seed": 42, "cache_prompt": True, "ignore_eos": True}
                attempt = 0
                proc = None
                primed = False
                while counts.get(key, 0) < REPS:
                    try:
                        if proc is None:
                            proc = start_server(variant, cap, gguf); primed = False
                        if not primed:
                            pbody = dict(body); pbody["n_predict"] = 1
                            c0, t0 = cpu_ticks(proc.pid), time.time()
                            r = http("/completion", pbody, timeout=3600)
                            rec = {"kind": "prime", "virgin": True, "variant": variant, "cap": cap, "ctx": ctx,
                                   "cpu_s": (cpu_ticks(proc.pid) - c0) / 100.0,
                                   "wall_s": time.time() - t0, "tokens_predicted": r.get("tokens_predicted"),
                                   "prompt_n": r.get("timings", {}).get("prompt_n")}
                            with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
                            primed = True
                            wbody = dict(body); wbody["n_predict"] = 32
                            for _ in range(5): http("/completion", wbody, timeout=3600)
                        c0, t0 = cpu_ticks(proc.pid), time.time()
                        r = http("/completion", body, timeout=3600)
                        cpu_s = (cpu_ticks(proc.pid) - c0) / 100.0
                        wall = time.time() - t0
                        blk = counts.get(key, 0)
                        rec = {"kind": "rep", "virgin": True, "variant": variant, "cap": cap, "ctx": ctx,
                               "block": blk, "cpu_s": cpu_s, "wall_s": wall,
                               "cpu_per_token": cpu_s / GEN, "wall_tok_s": GEN / wall,
                               "tokens_predicted": r.get("tokens_predicted"),
                               "srv_pred_ms": r.get("timings", {}).get("predicted_ms"),
                               "rss_kb_hwm": rss_kb(proc.pid)}
                        with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
                        counts[key] = counts.get(key, 0) + 1
                        attempt = 0
                        print(f"{variant} cap{cap} ctx{ctx} blk{blk}: cpu/tok={cpu_s/GEN:.4f}s walltok/s={GEN/wall:.2f} n={r.get('tokens_predicted')}", flush=True)
                    except Exception as e:
                        attempt += 1
                        print(f"  !! {variant} cap{cap} ctx{ctx}: {type(e).__name__} (attempt {attempt}); restarting server", flush=True)
                        if proc is not None: stop_server(proc); proc = None
                        if attempt >= 3:
                            print(f"  !! cell {variant} cap{cap} ctx{ctx} failed 3x; skipping (resume will retry)", flush=True)
                            break
                if proc is not None: stop_server(proc)
    print("MEASURE DONE")

if __name__ == "__main__":
    main()
