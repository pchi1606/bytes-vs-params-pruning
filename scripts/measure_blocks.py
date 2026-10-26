#!/usr/bin/env python3
"""Stage-1 measurement under SPEC v0.3 Amendment Round 3 (2026-10-24,
interim panel ADJUST/ADJUST). Phases, in order:
  FUNDING (D1): genuine temporal blocks for the funding contrast.
    Per (cap, ctx in {8192, 16384}): 5 blocks; each block measures
    V-width and V-mixed in ADJACENT fresh-process virgin cells
    (prime + 5x32 warmups + exactly 1 rep), order randomised within
    the block. Records carry "pair_block" and "sched": "funding".
  DIAGNOSTIC (D3): fresh-process remeasurement of V-depth@6144 and
    V-width/V-depth@16384, both caps, 5 reps per cell, each cell
    adjacent to the pair-partner's reference cell at the same
    (ctx, cap), order randomised. Records carry "diagnostic": true.
  POOL (D2): all remaining cells of V-depth-lite, V-depth-alt,
    V-KV, V-KV-lite, V-mixed (V-mixed's 8192/16384 cells excluded —
    they come from FUNDING) in one globally shuffled order, 5 reps
    per cell, virgin protocol.
All reps also record nivcsw_delta (server /proc status involuntary
context switches) and load1 (host loadavg at rep start) per D3.
Checkpoint/resume via results/stage1_measurements.jsonl, counting
each class (pool / funding / diagnostic) separately; pool counts
are the pre-amendment virgin counts (untagged virgin reps).

Amendment Round 4 (2026-10-25, interim panel Round 3, E1/E2):
  REPAIR phase between funding and diagnostic. A coordinator audit
  of every funding block against the six VM-restart times of
  2026-10-25 found three blocks whose pair members straddle a
  restart (D1 temporal-adjacency violation): cap2/8192 blk1
  (restart ~04:35 UTC), cap2/16384 blk0 (~09:06 UTC), cap4/16384
  blk0 (~13:34 UTC). Each is replaced IN FULL by one new adjacent
  pair of fresh-process virgin cells (order randomised), tagged
  "repair": true. Originals are preserved in the JSONL; the
  analysis pairs the replacement (explicit repair preference in
  analyze.py). Exclusion reason is the adjacency violation, never
  the measured value (E1/E3).
  Funding resume is now BLOCK-AWARE (E2): a funding block is
  skipped only when BOTH members are banked; an incomplete block
  is re-run whole, so no pair can again straddle an interruption.
"""
import json, os, random, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import (ROOT, OUT, CTXS, CAPS, GEN, start_server, stop_server,
                     http, cpu_ticks, rss_kb)

REPS = int(os.environ.get("REPS", "5"))
POOL_VARIANTS = ["V-depth-lite", "V-depth-alt", "V-KV", "V-KV-lite", "V-mixed"]
FUNDING_CTX = [8192, 16384]
N_BLOCKS = 5
# E1/E2 audited restart-split funding blocks, replaced whole in the
# REPAIR phase: (cap, ctx, pair_block). Audit basis: coordinator's
# verified banked-count brackets at each 2026-10-25 VM restart.
REPAIR_BLOCKS = [(2, 8192, 1), (2, 16384, 0), (4, 16384, 0)]

def seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xffffffff

def nivcsw(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("nonvoluntary_ctxt_switches"):
                    return int(line.split()[1])
    except Exception:
        pass
    return -1

def load_counts():
    pool, fund, diag, repairs = {}, {}, {}, {}
    if os.path.exists(OUT):
        for line in open(OUT):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("kind") != "rep" or not r.get("virgin"): continue
            if r.get("diagnostic"):
                k = (r["variant"], r["cap"], r["ctx"]); diag[k] = diag.get(k, 0) + 1
            elif "pair_block" in r:
                k = (r["variant"], r["cap"], r["ctx"], r["pair_block"]); fund[k] = fund.get(k, 0) + 1
                if r.get("repair"): repairs[k] = repairs.get(k, 0) + 1
            else:
                k = (r["variant"], r["cap"], r["ctx"]); pool[k] = pool.get(k, 0) + 1
    return pool, fund, diag, repairs

def run_cell(variant, cap, ctx, ids, want, tags, counts, key, block_value=None):
    """Fresh-process virgin cell: prime, 5 warmups, reps until the
    class counter for `key` reaches `want`. Returns reps recorded."""
    gguf = os.path.join(ROOT, "models", variant + "-Q4_K_M.gguf")
    body = {"prompt": ids, "n_predict": GEN, "temperature": 0,
            "seed": 42, "cache_prompt": True, "ignore_eos": True}
    got0 = counts.get(key, 0)
    attempt = 0
    proc = None
    primed = False
    while counts.get(key, 0) < want:
        try:
            if proc is None:
                proc = start_server(variant, cap, gguf); primed = False
            if not primed:
                pbody = dict(body); pbody["n_predict"] = 1
                c0, n0, t0 = cpu_ticks(proc.pid), nivcsw(proc.pid), time.time()
                r = http("/completion", pbody, timeout=3600)
                rec = {"kind": "prime", "virgin": True, "variant": variant, "cap": cap, "ctx": ctx,
                       "cpu_s": (cpu_ticks(proc.pid) - c0) / 100.0,
                       "wall_s": time.time() - t0, "tokens_predicted": r.get("tokens_predicted"),
                       "prompt_n": r.get("timings", {}).get("prompt_n"),
                       "srv_prompt_ms": r.get("timings", {}).get("prompt_ms"),
                       "nivcsw_delta": nivcsw(proc.pid) - n0,
                       "load1": os.getloadavg()[0]}
                rec.update(tags)
                with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
                primed = True
                wbody = dict(body); wbody["n_predict"] = 32
                for _ in range(5): http("/completion", wbody, timeout=3600)
            c0, n0, t0 = cpu_ticks(proc.pid), nivcsw(proc.pid), time.time()
            ld = os.getloadavg()[0]
            r = http("/completion", body, timeout=3600)
            cpu_s = (cpu_ticks(proc.pid) - c0) / 100.0
            wall = time.time() - t0
            blk = block_value if block_value is not None else counts.get(key, 0)
            rec = {"kind": "rep", "virgin": True, "variant": variant, "cap": cap, "ctx": ctx,
                   "block": blk, "cpu_s": cpu_s, "wall_s": wall,
                   "cpu_per_token": cpu_s / GEN, "wall_tok_s": GEN / wall,
                   "tokens_predicted": r.get("tokens_predicted"),
                   "srv_pred_ms": r.get("timings", {}).get("predicted_ms"),
                   "srv_prompt_ms": r.get("timings", {}).get("prompt_ms"),
                   "prompt_n": r.get("timings", {}).get("prompt_n"),
                   "rss_kb_hwm": rss_kb(proc.pid),
                   "nivcsw_delta": nivcsw(proc.pid) - n0, "load1": ld}
            rec.update(tags)
            with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
            counts[key] = counts.get(key, 0) + 1
            attempt = 0
            print(f"{variant} cap{cap} ctx{ctx} blk{blk} [{tags.get('sched', 'pool')}]: "
                  f"cpu/tok={cpu_s/GEN:.4f}s walltok/s={GEN/wall:.2f} n={r.get('tokens_predicted')} "
                  f"nivcsw={rec['nivcsw_delta']} load1={ld:.1f}", flush=True)
        except Exception as e:
            attempt += 1
            print(f"  !! {variant} cap{cap} ctx{ctx} [{tags.get('sched', 'pool')}]: "
                  f"{type(e).__name__} (attempt {attempt}); restarting server", flush=True)
            if proc is not None: stop_server(proc); proc = None
            if attempt >= 3:
                print(f"  !! cell {variant} cap{cap} ctx{ctx} failed 3x; skipping (resume will retry)", flush=True)
                break
    if proc is not None: stop_server(proc)
    return counts.get(key, 0) - got0

def main():
    prompts = json.load(open(os.path.join(ROOT, "results", "prompts.json")))
    pool, fund, diag, repairs = load_counts()
    phases = os.environ.get("PHASES", "funding,repair,diagnostic,pool").split(",")

    if "funding" in phases:
        print("== FUNDING blocks (D1) ==", flush=True)
        for cap in CAPS:
            for ctx in FUNDING_CTX:
                ids = prompts[str(ctx)]
                for b in range(N_BLOCKS):
                    order = ["V-width", "V-mixed"]
                    random.Random(seed(cap, ctx, b, "funding")).shuffle(order)
                    # E2 block-aware resume: skip only a COMPLETE block;
                    # an incomplete block is re-run whole (the banked
                    # member's fresh rep supersedes its stale one).
                    if all(fund.get((v, cap, ctx, b), 0) >= 1 for v in order):
                        for variant in order:
                            print(f"skip funding {variant} cap{cap} ctx{ctx} block{b}", flush=True)
                        continue
                    # G1: ONE durable attempt identifier per block
                    # attempt, computed ONCE before the variant loop
                    # (per-variant recomputation split blk4's pair
                    # across a2/a3 tags; fixed 2026-10-25).
                    n = max(fund.get((v, cap, ctx, b), 0) for v in order) + 1
                    for variant in order:
                        key = (variant, cap, ctx, b)
                        run_cell(variant, cap, ctx, ids, fund.get(key, 0) + 1,
                                 {"sched": "funding", "pair_block": b,
                                  "attempt": f"f{cap}-{ctx}-b{b}-a{n}"},
                                 fund, key, block_value=b)

    if "repair" in phases:
        print("== REPAIR blocks (E1/E2: restart-split pairs replaced whole) ==", flush=True)
        for cap, ctx, b in REPAIR_BLOCKS:
            ids = prompts[str(ctx)]
            order = ["V-width", "V-mixed"]
            random.Random(seed(cap, ctx, b, "repair")).shuffle(order)
            if all(repairs.get((v, cap, ctx, b), 0) >= 1 for v in order):
                for variant in order:
                    print(f"skip repair {variant} cap{cap} ctx{ctx} block{b}", flush=True)
                continue
            # G1: one attempt identifier per repair attempt,
            # computed ONCE before the variant loop.
            n = max(repairs.get((v, cap, ctx, b), 0) for v in order) + 1
            for variant in order:
                key = (variant, cap, ctx, b)
                run_cell(variant, cap, ctx, ids, repairs.get(key, 0) + 1,
                         {"sched": "funding", "pair_block": b, "repair": True,
                          "attempt": f"f{cap}-{ctx}-b{b}-repair{n}",
                          "repair_reason": "restart-split pair (D1 adjacency violation, E1/E2)"},
                         repairs, key, block_value=b)

    if "diagnostic" in phases:
        print("== DIAGNOSTIC remeasurement (D3) ==", flush=True)
        for ctx in (6144, 16384):
            for cap in CAPS:
                ids = prompts[str(ctx)]
                order = ["V-width", "V-depth"]
                random.Random(seed(ctx, cap, "diag")).shuffle(order)
                for variant in order:
                    key = (variant, cap, ctx)
                    if diag.get(key, 0) >= REPS:
                        print(f"skip diagnostic {variant} cap{cap} ctx{ctx}", flush=True); continue
                    run_cell(variant, cap, ctx, ids, REPS, {"sched": "diagnostic", "diagnostic": True},
                             diag, key)

    if "pool" in phases:
        print("== POOL interleaved (D2) ==", flush=True)
        cells = []
        for v in POOL_VARIANTS:
            for cap in CAPS:
                for ctx in CTXS:
                    if v == "V-mixed" and ctx in FUNDING_CTX: continue
                    if pool.get((v, cap, ctx), 0) >= REPS: continue
                    cells.append((v, cap, ctx))
        random.Random(seed("pool-v1")).shuffle(cells)
        print(f"pool cells remaining: {len(cells)}", flush=True)
        for variant, cap, ctx in cells:
            key = (variant, cap, ctx)
            run_cell(variant, cap, ctx, prompts[str(ctx)], REPS, {"sched": "pool"}, pool, key)
    print("MEASURE DONE")

if __name__ == "__main__":
    main()
