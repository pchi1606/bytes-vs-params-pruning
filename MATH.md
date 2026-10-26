# The Math — How Byte-Allocation Can Win ≥10% (2026-10-23)

Objective (unchanged): a task agent on the MacBook that does its work
using less CPU. Gemini's KILL is accepted for the design AS SPECIFIED
(4k context, depth-vs-width only). Below: where the arithmetic
actually leaves room, and the redesign it implies.

## 1. Corrected roofline (Llama-3.2-3B, Q4_K_M, fp16 KV cache)
Per decode step, DRAM traffic ≈ weight bytes streamed + KV read.
- Weights (unpruned): ~1.95 GB/token-step. Post-25%-cut, both arms: ~1.46 GB.
- KV per token: 2 × 28 layers × 8 KV-heads × 128 head-dim × 2 B = 0.115 MB.
  KV read per step at context c = 0.115 × c MB:
  c=4k → 0.47 GB | 8k → 0.94 | 16k → 1.88 | 32k → 3.76.
- KV share of total traffic: 4k ≈ 24% · 8k ≈ 39% · 16k ≈ 56% · 32k ≈ 72%.
Gemini used head-dim 64 (real: 128) and compared cache SIZE to one
step of weight traffic — corrected, the KV term is ~2× its estimate.

## 2. Arm traffic per decode step (GB), 25% param cut in both arms
Arm A = width allocation (KV untouched). Arm B = byte allocation.
| ctx | A | B: depth only (−7 layers) | B: depth + KV-heads 8→6 |
|-----|------|------|------|
| 4k  | 1.93 | 1.81 (−6%)  | 1.72 (−11%) |
| 8k  | 2.40 | 2.17 (−10%) | 1.99 (−17%) |
| 16k | 3.34 | 2.87 (−14%) | 2.52 (−25%) |
| 32k | 5.22 | 4.28 (−18%) | 3.67 (−30%) |
Assumes decode is bandwidth-bound, so CPU-seconds ≈ traffic ÷ BW.
Depth-only crosses the 10% bar at ~8k. Adding KV-head pruning (a pure
byte lever: KV heads carry traffic ∝ heads but few params) crosses it
even at 4k. A third lever, converting most layers to sliding-window
attention (window 2048, keep first/middle/last global — Hymba's
finding: 3 global layers recover recall, 3.8× cache cut; mask-only
SWA+sinks conversion is known to retain most quality), takes the 16k
saving to roughly −40%+. Each lever is prior art on its own; the claim
is the ALLOCATION comparison, not the levers.

## 3. Why the regime change is honest, not goalpost-moving
Agent workloads are long-context by construction: system prompt +
tool definitions + tool outputs + files routinely run 8–32k tokens.
The 4k design point was the artificial one. The claim becomes:
**byte-allocation wins where agents actually run (≥8k), and the
crossover context is itself a measured result.** A null at 4k is
reported, not hidden.

## 4. What can still kill it (stated before any run)
1. **KV quantization dilution:** if KV is stored q8_0/q4_0, KV traffic
   halves/quarters for BOTH arms and B's edge shrinks. Design must fix
   KV dtype (fp16) across arms, or report both — no silent choice.
2. **Compounded quality cliff:** depth cut + KV-head cut + SWA stack
   three quality risks. The pre-set tolerance (>2-pt mean task drop =
   breach) decides; levers are added one at a time (B1 depth, B2 +KV
   heads, B3 +SWA) so a breach is attributable.
3. **Prefill:** at long prompts prefill is compute-bound and roughly
   allocation-neutral (attention FLOPs at 16k ≈ 47% of prefill);
   CPU-seconds/task is reported split prefill/decode so a decode win
   can't hide behind, or be masked by, prefill.
4. **Bandwidth saturation at 2 threads** (Gemini's point): if both arms
   saturate DRAM at 2 threads, wall-time converges — but CPU-seconds
   (= user time, the metric) still separates them if B reaches the
   same tok/s at the lower cap. Measured, not assumed.

## 5. Revised design (SPEC v0.2 delta)
- Context sweep 4k / 8k / 16k (32k if time), thread caps 2 and 4.
- Arms: A (width), B1 (depth), B2 (depth + KV-heads 8→6); B3 (B2 +
  SWA hybrid) only if B2 clears quality. Same param cut ±1.5%,
  identical recovery budget, fp16 KV fixed.
- Decision rule v2: PURSUE if any B arm beats A by ≥10% CPU-sec/task
  at equal quality at BOTH 8k and 16k; 4k is reported, not gating.
  Otherwise KILL stands and the crossover/null result is the report.

Status: pilot remains ON HOLD. Revival = his decision on v0.2.
