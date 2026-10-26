# STAGE 1 REPORT — Bytes-vs-Params Pruning (SPEC v0.3)

**Status: FROZEN stage-1 report, 2026-10-26.** Measurement complete
(driver printed MEASURE DONE after funding → repair → diagnostic →
pool phases; 521 virgin rep records banked). Frozen analysis:
`scripts/analyze.py` (G1-amended 2026-10-25, manifest-driven) →
`results/stage1_analysis.json`. Funding-attempt manifest:
`results/funding_attempts_manifest.json` (H3 completeness
assertions PASSED at MEASURE DONE: 20 retained pairs, 5 per
funding cell, 3 repaired blocks, 40 distinct rows, none pending).

## Verdicts

### Funding hurdle (D1) — NOT MET
Criterion (frozen, SPEC §B + D1): the predeclared byte candidate
V-mixed beats the parameter-count candidate V-width with the upper
95% CI of the paired log-time ratio below 0.90 at BOTH 8192 and
16384, in 5 valid temporally-adjacent fresh-process block pairs
per (cap, ctx). Results (manifest-retained pairs only):

| cap | ctx | n pairs | ratio (V-mixed/V-width) | 95% CI | upper < 0.90? |
|-----|-----|---------|------------------------|--------|---------------|
| 2 | 8192 | 5 (blk1 repaired) | 0.917 | [0.824, 1.020] | NO |
| 2 | 16384 | 5 (blk0 repaired) | 0.986 | [0.934, 1.041] | NO |
| 4 | 8192 | 5 | 0.818 | [0.795, 0.842] | YES |
| 4 | 16384 | 5 (blk0 repaired) | 0.817 | [0.609, 1.096] | NO |

Cap2 is SETTLED-FAILED at both contexts (H1). Cap4 passes at 8192
only and fails at 16384 on precision. No cap satisfies the
conjunctive criterion: **the funding hurdle is not met.** The
traffic model's predicted ratios (0.771 at 8192, 0.673 at 16384)
were systematically optimistic — observed ratios are higher
(less advantage) at every cell, decisively so at cap2. Per E5,
the 0.771 prediction was specific to cap2/8192, where the
observed ratio is 0.917; its closeness to the cap4/8192 observed
ratio (0.818) does not validate it. Per E5 the cap pattern is an
exploratory interaction observation only (caps compared at
matched contexts; frozen models unchanged).

### Scientific verdict (E7) — PRACTICALLY NEGATIVE
The funding hurdle alone does not assign this verdict (E7); it is
decided by the frozen C≥4096 leave-one-variant-out (LOVO)
prediction contest, ΔAIC, and prospective-contrast agreement on
the full eligible data (451 primary reps; V-KV-lite excluded per
D4). The byte-traffic model does NOT win the contest at either
cap:

| cap | params LOVO rel err | traffic LOVO rel err | ΔAIC (P−B) | traffic wins? |
|-----|--------------------|----------------------|------------|---------------|
| 2 | 0.743 | 0.689 | +2.21 | NO (ΔAIC « 10) |
| 4 | 0.284 | 0.538 | −19.11 | NO (params wins both) |

Sensitivity (V-KV-lite included, 155/156 reps): same outcome
(cap2 ΔAIC +3.37; cap4 ΔAIC −19.56, params LOVO better). The
prospective contrast agrees with the contest: V-mixed's advantage
over V-width is real at cap4 (~0.82) but absent at cap2
(0.92/0.99, CIs include 1) and everywhere smaller than the
traffic model predicts. The pre-registered stage-1 PASS
condition therefore fails on adequately precise data (tight CIs
at the decisive cells; the cap4 contest loss is decisive, not a
power failure): **stage 1 is PRACTICALLY NEGATIVE for the claim
that per-token byte traffic out-predicts parameter count for CPU
decode latency in this design. Stage 2 does NOT unlock.** This
report is the end of the track unless the user rules otherwise
at the frozen-report → manuscript gate, which remains his.

Layer-count coefficients (frozen models): cap2 — params model
L coef +0.00125, traffic model L coef +0.00771; cap4 — params
−0.00334, traffic −0.00323. In-sample R²: cap2 P 0.412 / B 0.413;
cap4 P 0.608 / B 0.542. (For scale only: the unscaled raw-byte
predictor's held-out relative error is 4.34 (cap2) / 4.97
(cap4) — the fitted models, not the raw formula, are the
contestants; see analyze.py notes.)

## Variants (all ~25% pruned from Llama-3.2-3B; param spread 1.84%)

| variant | layers L | Σ kv-heads | params P | tensor-payload bytes | GGUF file bytes | allocation |
|---------|----------|-----------|----------|---------------------|-----------------|------------|
| V-width | 28 | 224 | 2,442,046,464 | 1,552,472,320 | 1,560,309,888 | width (FFN) trim only |
| V-depth | 21 | 168 | 2,408,973,312 | 1,526,016,256 | 1,533,850,048 | 7 whole layers dropped |
| V-depth-lite | 24 | 192 | 2,413,710,336 | 1,530,366,208 | 1,538,201,632 | 4 layers dropped |
| V-depth-alt | 21 | 168 | 2,408,973,312 | 1,526,016,256 | 1,533,850,048 | 7 layers, alternate positions |
| V-KV | 28 | 112 | 2,398,006,272 | 1,527,699,712 | 1,535,537,280 | KV heads halved (GQA 8→4) |
| V-KV-lite | 28 | 168 | 2,442,046,464 | 1,549,633,792 | 1,557,471,360 | KV heads 8→6 — EXPORT INVALID, see below |
| V-mixed | 23 | 92 | 2,419,995,648 | 1,539,152,128 | 1,546,987,008 | predeclared byte-greedy mix (5 layers + KV trim) |

Tensor-payload bytes and GGUF file bytes are reported separately
per D6 (file bytes include headers/metadata). V-mixed-shallow
was DROPPED before measurement (D6): its 8k/16k byte-greedy
allocations are structurally identical to V-mixed. Base-model
deviation on record: the gated Meta repo was substituted with
the cached unsloth/Llama-3.2-3B-Instruct snapshot (same weights).

## Cell means (CPU s/token, virgin reps; n, mean, SD, ±95% CI, wall tok/s)

| variant | cap | ctx | n | mean | SD | CI95 | wall tok/s |
|---|---|---|---|---|---|---|---|
| V-KV | 2 | 512 | 5 | 0.1570 | 0.0052 | ±0.0065 | 8.45 |
| V-KV | 2 | 2048 | 5 | 0.2006 | 0.0124 | ±0.0153 | 6.84 |
| V-KV | 2 | 4096 | 5 | 0.2617 | 0.0106 | ±0.0132 | 5.29 |
| V-KV | 2 | 6144 | 5 | 0.3235 | 0.0069 | ±0.0086 | 4.34 |
| V-KV | 2 | 8192 | 5 | 0.3797 | 0.0109 | ±0.0135 | 3.78 |
| V-KV | 2 | 16384 | 5 | 0.9178 | 0.1132 | ±0.1405 | 1.25 |
| V-KV | 4 | 512 | 5 | 0.1473 | 0.0012 | ±0.0015 | 9.68 |
| V-KV | 4 | 2048 | 5 | 0.1821 | 0.0014 | ±0.0017 | 4.14 |
| V-KV | 4 | 4096 | 5 | 0.2449 | 0.0091 | ±0.0113 | 4.16 |
| V-KV | 4 | 6144 | 5 | 0.3032 | 0.0151 | ±0.0187 | 4.50 |
| V-KV | 4 | 8192 | 5 | 0.3555 | 0.0053 | ±0.0066 | 3.96 |
| V-KV | 4 | 16384 | 5 | 0.6694 | 0.0165 | ±0.0205 | 2.04 |
| V-KV-lite | 2 | 512 | 5 | 0.1598 | 0.0113 | ±0.0141 | 8.82 |
| V-KV-lite | 2 | 2048 | 5 | 0.2046 | 0.0067 | ±0.0083 | 6.48 |
| V-KV-lite | 2 | 4096 | 5 | 0.2692 | 0.0133 | ±0.0165 | 4.87 |
| V-KV-lite | 2 | 6144 | 5 | 0.3230 | 0.0088 | ±0.0109 | 4.33 |
| V-KV-lite | 2 | 8192 | 5 | 0.3784 | 0.0097 | ±0.0120 | 3.73 |
| V-KV-lite | 2 | 16384 | 5 | 0.7263 | 0.0212 | ±0.0263 | 1.73 |
| V-KV-lite | 4 | 512 | 5 | 0.1508 | 0.0014 | ±0.0017 | 9.37 |
| V-KV-lite | 4 | 2048 | 5 | 0.1909 | 0.0017 | ±0.0021 | 7.39 |
| V-KV-lite | 4 | 4096 | 5 | 0.2500 | 0.0028 | ±0.0035 | 4.73 |
| V-KV-lite | 4 | 6144 | 5 | 0.3053 | 0.0023 | ±0.0029 | 4.29 |
| V-KV-lite | 4 | 8192 | 5 | 0.3547 | 0.0012 | ±0.0015 | 3.94 |
| V-KV-lite | 4 | 16384 | 5 | 0.6105 | 0.0044 | ±0.0055 | 2.21 |
| V-depth | 2 | 512 | 5 | 0.1565 | 0.0113 | ±0.0140 | 8.76 |
| V-depth | 2 | 2048 | 5 | 0.2076 | 0.0070 | ±0.0086 | 6.41 |
| V-depth | 2 | 4096 | 5 | 0.2544 | 0.0013 | ±0.0016 | 5.67 |
| V-depth | 2 | 6144 | 5 | 0.4738 | 0.1321 | ±0.1640 | 2.38 |
| V-depth | 2 | 8192 | 5 | 0.4150 | 0.0266 | ±0.0330 | 3.38 |
| V-depth | 2 | 16384 | 5 | 1.1865 | 0.0913 | ±0.1134 | 0.42 |
| V-depth | 4 | 512 | 5 | 0.1388 | 0.0010 | ±0.0012 | 3.38 |
| V-depth | 4 | 2048 | 5 | 0.1827 | 0.0013 | ±0.0016 | 2.69 |
| V-depth | 4 | 4096 | 5 | 0.2440 | 0.0020 | ±0.0025 | 2.32 |
| V-depth | 4 | 6144 | 5 | 0.4998 | 0.0051 | ±0.0063 | 1.27 |
| V-depth | 4 | 8192 | 5 | 0.3721 | 0.0039 | ±0.0049 | 1.49 |
| V-depth | 4 | 16384 | 5 | 0.9835 | 0.5259 | ±0.6529 | 0.63 |
| V-depth-alt | 2 | 512 | 5 | 0.1678 | 0.0154 | ±0.0191 | 7.58 |
| V-depth-alt | 2 | 2048 | 5 | 0.2154 | 0.0132 | ±0.0164 | 6.33 |
| V-depth-alt | 2 | 4096 | 5 | 0.2614 | 0.0035 | ±0.0044 | 5.50 |
| V-depth-alt | 2 | 6144 | 5 | 0.3455 | 0.0106 | ±0.0132 | 4.03 |
| V-depth-alt | 2 | 8192 | 5 | 0.3970 | 0.0060 | ±0.0075 | 3.41 |
| V-depth-alt | 2 | 16384 | 5 | 0.6190 | 0.0085 | ±0.0105 | 2.27 |
| V-depth-alt | 4 | 512 | 5 | 0.1465 | 0.0023 | ±0.0029 | 8.27 |
| V-depth-alt | 4 | 2048 | 5 | 0.1807 | 0.0029 | ±0.0036 | 3.87 |
| V-depth-alt | 4 | 4096 | 5 | 0.2439 | 0.0019 | ±0.0023 | 3.37 |
| V-depth-alt | 4 | 6144 | 5 | 0.3086 | 0.0031 | ±0.0038 | 2.35 |
| V-depth-alt | 4 | 8192 | 5 | 0.3789 | 0.0078 | ±0.0097 | 3.54 |
| V-depth-alt | 4 | 16384 | 5 | 0.5924 | 0.0050 | ±0.0062 | 2.01 |
| V-depth-lite | 2 | 512 | 5 | 0.1461 | 0.0032 | ±0.0039 | 9.84 |
| V-depth-lite | 2 | 2048 | 5 | 0.2365 | 0.0129 | ±0.0160 | 5.37 |
| V-depth-lite | 2 | 4096 | 5 | 0.5468 | 0.0234 | ±0.0291 | 1.32 |
| V-depth-lite | 2 | 6144 | 5 | 0.5202 | 0.2339 | ±0.2904 | 2.80 |
| V-depth-lite | 2 | 8192 | 5 | 0.4574 | 0.0126 | ±0.0156 | 2.85 |
| V-depth-lite | 2 | 16384 | 5 | 1.3763 | 0.9376 | ±1.1640 | 1.16 |
| V-depth-lite | 4 | 512 | 5 | 0.1481 | 0.0018 | ±0.0022 | 9.16 |
| V-depth-lite | 4 | 2048 | 5 | 0.1980 | 0.0008 | ±0.0010 | 6.80 |
| V-depth-lite | 4 | 4096 | 5 | 0.2687 | 0.0015 | ±0.0019 | 5.02 |
| V-depth-lite | 4 | 6144 | 5 | 0.3318 | 0.0033 | ±0.0041 | 2.37 |
| V-depth-lite | 4 | 8192 | 5 | 0.4137 | 0.0030 | ±0.0038 | 3.36 |
| V-depth-lite | 4 | 16384 | 5 | 0.6706 | 0.0075 | ±0.0093 | 2.02 |
| V-mixed | 2 | 512 | 5 | 0.1430 | 0.0065 | ±0.0081 | 9.99 |
| V-mixed | 2 | 2048 | 5 | 0.2044 | 0.0214 | ±0.0265 | 6.40 |
| V-mixed | 2 | 4096 | 5 | 0.2301 | 0.0041 | ±0.0051 | 6.09 |
| V-mixed | 2 | 6144 | 5 | 0.2800 | 0.0142 | ±0.0176 | 5.17 |
| V-mixed | 2 | 8192 | 6 | 0.4155 | 0.1076 | ±0.1130 | 3.19 |
| V-mixed | 2 | 16384 | 6 | 0.6548 | 0.0340 | ±0.0357 | 2.07 |
| V-mixed | 4 | 512 | 5 | 0.1406 | 0.0012 | ±0.0014 | 10.00 |
| V-mixed | 4 | 2048 | 5 | 0.1754 | 0.0019 | ±0.0024 | 8.00 |
| V-mixed | 4 | 4096 | 5 | 0.2187 | 0.0034 | ±0.0042 | 6.29 |
| V-mixed | 4 | 6144 | 5 | 0.2555 | 0.0041 | ±0.0051 | 3.76 |
| V-mixed | 4 | 8192 | 5 | 0.3097 | 0.0036 | ±0.0045 | 4.28 |
| V-mixed | 4 | 16384 | 10 | 0.6248 | 0.1060 | ±0.0758 | 1.85 |
| V-width | 2 | 512 | 5 | 0.1549 | 0.0115 | ±0.0143 | 9.08 |
| V-width | 2 | 2048 | 5 | 0.1915 | 0.0017 | ±0.0021 | 7.51 |
| V-width | 2 | 4096 | 5 | 0.2573 | 0.0124 | ±0.0154 | 5.56 |
| V-width | 2 | 6144 | 5 | 0.2985 | 0.0068 | ±0.0084 | 4.85 |
| V-width | 2 | 8192 | 12 | 0.5584 | 0.5575 | ±0.3380 | 3.18 |
| V-width | 2 | 16384 | 11 | 0.8183 | 0.2196 | ±0.1475 | 1.46 |
| V-width | 4 | 512 | 5 | 0.1519 | 0.0018 | ±0.0022 | 7.87 |
| V-width | 4 | 2048 | 5 | 0.1921 | 0.0010 | ±0.0013 | 7.30 |
| V-width | 4 | 4096 | 5 | 0.2501 | 0.0015 | ±0.0019 | 5.10 |
| V-width | 4 | 6144 | 5 | 0.3097 | 0.0009 | ±0.0011 | 4.01 |
| V-width | 4 | 8192 | 10 | 0.3726 | 0.0089 | ±0.0064 | 2.85 |
| V-width | 4 | 16384 | 11 | 0.7659 | 0.3217 | ±0.2161 | 1.81 |
Cells with n > 5 are V-width/V-mixed at funding contexts: their
funding and repair reps are ordinary eligible virgin reps for
cell means under the frozen D5 definitions (only the D1 paired
contrast is restricted to manifest-retained block pairs).
Prefill (prime) CPU means per cell are in the analysis JSON.

## Saturation and thread caps (E6, G4)
Caps are llama.cpp thread counts (-t 2 / -t 4) under a
`taskset -c 0,1` pin to 2 cores: "cap4" is 4 threads on 2 cores,
NOT 4 cores, and no cap4 advantage is attributed to more cores.
Wall-tok/s saturation at 16384 (cap4 ≤ cap2): V-depth-alt
(2.27 → 2.01 tok/s) and V-mixed (2.07 → 1.85 tok/s) are flagged
saturated; all other variants are not. Cap comparisons for the
two flagged variants at 16384 are saturation-regime and are not
read as clean cap effects (E5/E6).

## Diagnostics (D3) and the load-era caveat
Diagnostic remeasurement (fresh virgin cells, never in primary
fits) vs the original pool cell means:

| cell | diag mean | original mean | ratio |
|------|-----------|---------------|-------|
| V-depth cap2 6144 | 0.3275 | 0.4738 | 0.691 |
| V-depth cap4 6144 | 0.3113 | 0.4998 | 0.623 |
| V-depth cap2 16384 | 0.6758 | 1.1865 | 0.570 |
| V-depth cap4 16384 | 0.5893 | 0.9835 | 0.599 |
| V-width cap2 6144 | 0.3564 | 0.2985 | 1.194 |
| V-width cap4 6144 | 0.3183 | 0.3097 | 1.028 |
| V-width cap2 16384 | 0.7196 | 0.8183 | 0.879 |
| V-width cap4 16384 | 0.6050 | 0.7659 | 0.790 |

Coordinator-3's E4/G3 adjudication (recorded in
STAGE1-RUN-STATUS.md before the pool proceeded): the original
V-depth pool cells were measured 2026-10-24 in the
highest-contention window (host load 8–10 from other tracks);
the same virgin fresh-process protocol in both eras, with
per-rep load1/nivcsw recorded, makes this a load-era contention
effect on the early cells, NOT an active measurement defect —
no driver repair existed or was permitted (remeasuring valid
cells is barred by E3; fit changes by G4). Under matched fresh
measurement the Round-1 "V-depth anomalously slow" puzzle does
not reproduce (diagnostic V-depth is the fastest variant at
16384). The discrepancy is carried as a load-variance caveat on
the early pool cells; the frozen primary fits use the originals
as designed, and the Gate-3 panel should weigh this caveat —
it biases, if anything, against the depth allocations, not
against the parameter model in the contest (V-width, the
params-side reference, agrees across eras at 6144 and runs
0.79–0.88 at 16384).

## V-KV-lite — functional check FAIL (D4)
Teacher-forced logits check (144 positions): base dNLL −0.0098,
KL 0.192, top-1 0.958; V-KV-lite dNLL +3.761, KL 18.79, top-1
0.000. Frozen D4 rule (pass iff dNLL ≤ base + 0.05 AND mean KL ≤
2× base AND top-1 ≥ base − 0.10, i.e. thresholds −0.0098+0.05,
0.384, 0.858): **FAIL on all three.** Root cause: a genuine GGUF
export bug — the converter laid out Q/K tensors under a false
24/8 head configuration before an in-place metadata patch; it is
not a quantisation effect. Disposition (D4/E8/F5/G5, unchanged
through every round): excluded from the primary prediction fit,
included as sensitivity; its timings were collected on schedule
and are retained, but they are NOT evidence for a functionally
valid pruning candidate. No rebuild was performed.

## Anomalies, incidents, and amendment register
- **Slot-history contamination (pre-§C):** multi-context sessions
  on one server returned 1.62–1.71 vs 0.62–0.64 CPU-s/token for
  identical cells → §C3 virgin-process protocol (one fresh
  llama-server per cell); 30 pre-virgin reps EXCLUDED from all
  analysis, retained in the JSONL unflagged (analyze.py counts
  them as n_excluded_previrgin_reps = 30).
- **Amendments §C (runtime):** C1 reps 10→5; C3 virgin protocol;
  C4 prefill-only primes; C5 restart-on-error/timeout (3600 s per
  request, max 3 attempts, checkpoint resume). Context 12288
  dropped as a runtime amendment (D6 corrects the record: the
  earlier citation of a spec provision for it was wrong).
- **Round 2 (§D1–D7):** D1 funding blocks (whole adjacent pairs,
  randomised order); D2 randomised interleaved pool via
  measure_blocks.py; D3 diagnostics; D4 V-KV-lite rule; D5 frozen
  analysis definitions in analyze.py; D6 record corrections
  (incl. tensor-payload vs file bytes; V-mixed-shallow dropped);
  D7 correction of the Gemini seat's KV-bytes error (V-depth
  reads ~25% less KV traffic: 21 vs 28 layers × 8 heads).
- **Round 3 (§E1–E8):** E1/E2 whole-block replacement of
  restart-split pairs with block-aware resume and a full funding
  audit (audit found exactly 3 split blocks); E3 no
  outcome-driven reruns; E4 phase order + diagnostics first;
  E5 cap-pattern discipline; E6 thread-cap wording; E7
  hurdle/verdict separation; E8 V-KV-lite unchanged.
- **Round 4 (F1–F5):** F1 invalid-block exclusion in analyze.py
  pending repairs (superseded by G1's manifest); F2 no covariate
  adjustment; F3 phase order stands; F4 interim cap4/16384
  wording; F5 V-KV-lite stands.
- **Round 4 supplement (G1–G5):** G1 attempt-level manifest
  (immutable row ids + JSONL md5 gate; silent per-variant
  last-wins PROHIBITED; analyze.py rewritten manifest-driven,
  prior code/output preserved as analyze_f1_superseded.py and
  stage1_analysis_STALE-2026-10-25-0210.json); G2 interim
  language corrections (applied in SPEC §E7 and status file);
  G3 schedule completion order; G4 frozen analyses on audited
  data only; G5 V-KV-lite stands.
- **Round 5 (H1–H5):** H1 cap2 hurdle settled-failed language;
  H2 blk4 split-tag merge accepted conditionally on adjacency
  evidence (supplied: JSONL rows 291–294 = prime/rep/prime/rep
  consecutive inside one driver run, pid 9737, 18:44–~19:10 UTC,
  plus verbatim log lines preserved in STAGE1-RUN-STATUS.md);
  H3 manifest rebuilds with completeness assertions (passed at
  post-repair and at MEASURE DONE); H4 blk0 repair2 whole-block
  rule (repair1's V-mixed singleton 0.5533 explicitly superseded,
  never combined); H5 diagnostics NOT redirected to V-KV-lite.
- **Coordinator split-tag bug (own, found + fixed 2026-10-25):**
  the first attempt-tag build recomputed the attempt id inside
  the per-variant loop, splitting cap4/16384 blk4's genuine pair
  across tags a2/a3; fixed (id computed once per block, funding
  and repair loops); the pair is merged in the manifest with
  provenance (MERGE_SPLIT_TAG), never on performance.
- **Funding attempt accounting (final manifest):** valid 17,
  invalid_restart_split 3 (cap2/8192 blk1, cap2/16384 blk0,
  cap4/16384 blk0 — all repaired), incomplete_superseded 5,
  incomplete_pending 0, repaired_valid 3, lost_interrupted 1.
- **VM restarts ×18** during the run (2026-10-25/26), each
  costing at most the in-flight cell; checkpoint/block-aware
  resume verified after each. Coordinators 1–2 also died of
  runtime restarts; coordinator-3 supervised from 2026-10-25.
- **C5 incident:** V-KV cap2/16384 pool cell hit
  RemoteDisconnected (server died mid-cell); server restarted,
  cell completed on retry. One server wedge earlier in the run
  was resolved by the same protocol; the 3600 s timeout never
  fired on a completed long request.
- **Export-check insensitivity:** the original export check
  passed V-KV-lite's GGUF despite the layout bug — the
  teacher-forced logits check (D4) is the sensitive instrument.
  HF-side validation ran in memory because transformers 5.20.2's
  validator cannot reload the patched config; GGUF phases used
  scripts/check_kvlite_gguf_only.py. One OOM kill occurred during
  the combined logits-check run (7.9 GB VM) — rerun split-phase.
- **Disk-full incidents** earlier in the track (browser upload
  spool starvation at 97% disk, 2026-10-24) were resolved by the
  disk-hygiene pass; no measurement data was lost.
- **Load variance:** host load ranged ~2.5–10 across the run
  (other tracks' work, never touched). CPU-s/token is the frozen
  primary metric; per-rep load1 and nivcsw are recorded for
  every rep. No covariate adjustment (F2/G4).

## Independent review history (no KILL/reject language in any round)
Gate 1: Gemini KILL of the superseded 4k design (accepted;
design reframed to v0.3 long-context), Perplexity ADJUST.
Gate 2: ADJUST/ADJUST, closed. Interim rounds 1–5 (rolling 4h):
ADJUST throughout — Perplexity (Deep Research; Round 5 in
Search mode after a Max-upgrade dialog, recorded honestly),
Gemini (Round 4 both external seats approval-blocked and owed;
Round 4 supplement: Perplexity landed, Gemini signed out;
Round 5: Gemini blocked on a holder-only phone QR verification
for aipeppers01@gmail.com — seat still OWED at Gate 3; do not
switch accounts), fresh in-house reviewers concurring. All
adopted fixes are recorded in PANEL-INTERIM-2026-10-24.md and
REVIEW-FIXES-INBOX.md and were applied by the coordinators.

## Successor instructions (standing, user's orders)
Gates run autonomously; fix reviewer issues on the agent's
recommendation; alert the user IMMEDIATELY on any
independent-reviewer KILL or weak-reject/desk-reject/reject
language; after any fix, re-review the FULL artifact with FRESH
reviewers until comments go stale. Stage 1 verdict is
PRACTICALLY NEGATIVE, so stage 2 does not start. The
frozen-report → manuscript decision is the USER's. Do not raise
model naming (deferred by the user until results are in — they
now are; naming remains his call). models/V-KV-lite-hf/
(regenerable) was deleted after this report was written.
