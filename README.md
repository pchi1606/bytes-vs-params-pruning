# When Do Parameter Budgets Misprice CPU Inference? A Matched-Budget Test of Byte-Traffic Pruning Allocation

This repository holds the code, frozen preregistration, raw measurements, and
manuscript for a matched-budget pruning study on Llama-3.2-3B. Seven variants
were pruned by ~25% from the same search space (layers, FFN channels, KV
groups), differing only in the cost model used to allocate the cut: parameter
count, per-token byte traffic (packed weight bytes plus KV-cache read bytes at
the occupied context), or measured latency. All variants were timed as GGUF
Q4_K_M models under fixed CPU thread caps (2 and 4 threads, pinned to 2
cores) at occupied contexts from 512 to 16384 tokens, 521 virgin reps in
total.

**Result (bounded negative).** The predeclared byte-traffic candidate did not
beat the parameter-count candidate: the funding hurdle (upper 95% CI of the
paired log-time ratio below 0.90 at both 8192 and 16384) was not met at either
thread cap (cap 2 settled-failed, upper bounds 1.020 at 8192 and 1.041 at
16384; cap 4 passed at 8192 only). In the held-out prediction contest the
traffic model did not beat the parameter model (leave-one-variant-out
ΔAIC +2.21 at cap 2, well under the predeclared 10-point bar; at cap 4 the
parameter model won outright, ΔAIC −19.11). Within this matched-budget CPU
regime, an analytical weight-plus-KV traffic model did not improve held-out
latency prediction or allocation selection over parameter count. See
`STAGE1-REPORT.md` for the frozen verdict and full anomaly record.

## Layout

- `SPEC-v0.3.md` — frozen preregistration (criteria, amendments C/D/E/H
  series recorded in place); `SPEC.md` — the original v0.1 spec;
  `MATH.md` — the corrected traffic arithmetic the design rests on.
- `scripts/` — variant construction, measurement, and analysis code.
- `results/` — raw measurements (`stage1_measurements.jsonl`, 521 virgin
  reps), the frozen analysis (`stage1_analysis.json`), the funding-attempt
  manifest, packed-byte accounting, the V-KV-lite logits check, and the
  selection-regret computation.
- `STAGE1-REPORT.md` — the frozen stage-1 report (verdicts, cell statistics,
  all incidents and amendments).
- `PANEL-INTERIM-2026-10-24.md` — the independent review record (gates,
  rolling rounds, API literature panel).
- `manuscript/` — the bounded-negative manuscript (`paper.tex`, `paper.md`,
  `paper.pdf`), its test-23 battery record (`BATTERY.md`) and review panel
  record (`PANEL-23B.md`); `manuscript/emc2/` holds the EMC² workshop
  (AAAI 2027) version.

## Reproduction entry points

- `scripts/measure_blocks.py` — stage-1 measurement driver (phases:
  funding blocks → diagnostics → interleaved pool; checkpoint/resume via
  `results/stage1_measurements.jsonl`). Requires the variant GGUFs and a
  llama.cpp server build; model weights are not distributed here (see
  `scripts/build_all.sh` and the prune scripts for how variants were built
  from the public Llama-3.2-3B snapshot).
- `scripts/analyze.py` — the frozen analysis: reads the measurements and
  reproduces `results/stage1_analysis.json` (funding contrast, LOVO
  prediction contest, ΔAIC) under the definitions frozen in its header.

## Licence

Not yet set.
