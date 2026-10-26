# SPEC v0.3 — Preregistration (2026-10-23, revival under Gate-1 ADJUST)

Supersedes SPEC.md (v0.1) and the v0.2 delta in MATH.md. Frozen before
any stage-1 run. Base: unsloth/Llama-3.2-3B-Instruct (locally cached
snapshot; meta-llama repo gated — recorded deviation, same weights).

## Reframed question
When do parameter budgets misprice CPU inference? Specifically: does a
**weights + KV byte cost model** predict and allocate CPU decode cost
better than **parameter count**, and does it approach the allocation a
**direct-latency** cost model would choose (ZipLM/HAPE-style)?

## Panel fixes baked in (from PANEL-GATE1.md)
1. **Common search space.** All allocators draw from the SAME unit
   space: removable layers, FFN intermediate channels, KV groups (with
   their Q groups). Allocators differ ONLY in cost model. The original
   width-only arm survives as a practical reference (V-width), not the
   sole scientific control.
2. **Marginal-cost allocation.** Byte allocation is defined on marginal
   savings ΔB(u,S) = B(M,S) − B(M\u,S) at the design context, not on
   original module cost, and NOT hard-coded depth-heavy: the cost
   model must produce the allocation (V-mixed).
3. **Occupied context** (actually populated KV tokens) is the x-axis,
   not context capacity.
4. **CPU-seconds-per-success** in stage 2 counts failures/retries in
   the numerator. Stage 1 primary metric: CPU user+sys seconds per
   generated token (process time via /usr/bin/time, load-only baseline
   subtracted), at fixed occupied contexts.
5. **Packed bytes reported** for every variant (actual GGUF file
   bytes), never just the quant label.
6. **Recovery equality** (stage 2, if reached): identical training
   tokens, optimizer steps, LoRA rank/targets, and reported trainable
   parameter counts for every arm; no-recovery reported as a separate
   condition.
7. **Quality (stage 2):** paired non-inferiority vs the param
   allocator, margin and task count set from stage-2 pilot variance —
   not a fixed "≤2 points on 40 binary tasks" rule.

## Stage 1 — mechanism test (THE kill point; this preregistration's scope)
No recovery training, no task suite. Question: does the logical-traffic
model predict measured decode cost, and does any byte-favored allocation
actually beat the width reference at long occupied context?

### Variants (each ~25% param cut, final params pairwise within ±1%)
Common unit space: layers / FFN channels / KV groups.
- **V-width** — param-share reference: uniform FFN-intermediate +
  Q-head pruning; KV groups untouched.
- **V-depth** — plain depth reference: remove 7 lowest Block-Influence
  layers (fresh calibration) PLUS an explicitly specified FFN trim
  sized to land the total cut at 25% (see Amendment §1: 7 layers alone
  are 21.93%, not 25%). Recorded as depth+trim, not pure depth.
- **V-KV** — KV groups 8→4 in all layers + FFN trim to match.
- **V-mixed** — byte-proportional allocation at 16k design context
  (marginal decode bytes: packed weight bytes + KV read at 16k):
  depth + KV-group + FFN units combined per ΔB ranking.
Each variant: physically resized tensors → loadable HF checkpoint →
param count + config verified → GGUF Q4_K_M → GGUF load + sane
generation verified. Any variant that cannot be built/exported is
documented and dropped; the rest proceed.

### Measurement (llama.cpp, this VM, x86)
- Occupied contexts: 512 / 4096 / 8192 / 16384 (fixed neutral filler
  prompt occupying the cache), fixed 128-token generations, temp 0,
  fixed seed.
- Thread caps: 2 and 4 (llama.cpp -t). KV dtype fp16 (fixed; MATH.md
  kill-risk #1 — no silent KV quantization).
- ≥3 repeats per cell; /usr/bin/time user+sys CPU; tok/s; peak RSS;
  machine settings recorded (2-core VM, governor, llama.cpp build).
- Prompt-cache used so repeats measure decode only; a load-only
  (-n 0) run per cell is subtracted as overhead baseline.

### Pre-committed analysis
- Two prediction models for CPU-sec/token across all variant×context
  cells (fit per thread cap): (P) parameter count; (B) logical traffic
  = packed weight bytes + KV-read bytes at occupied context. Report
  Spearman ρ and OLS R²/MAE for both, including a context-interaction
  check for P (params + ctx + params×ctx).
- Headline contrast: best byte-favored variant (V-depth / V-KV /
  V-mixed) vs V-width CPU-sec/token at 8k and 16k.

### Stage-1 PASS / FAIL (both required for PASS)
(a) Traffic model clearly beats param model at prediction (higher ρ
    AND higher R² / lower MAE at both thread caps), AND
(b) ≥10% CPU-sec/token advantage for the best byte-favored variant
    over V-width at BOTH 8k and 16k occupied contexts.
Otherwise stage 1 FAILS: the null is written up in STAGE1-REPORT.md
and the track does NOT proceed to stage 2. No metric shopping, no
post-hoc context picking.

## Stage 2 (only if stage 1 PASSes; separately preregistered then)
Recovery training under fix #6, task suite under fixes #4/#7,
direct-latency allocator (T) materialized as a comparator arm.

## Budget
$0 spend. VM CPU only. Checkpointed throughout.

---

# Amendment — Gate 2 (2026-10-23, Perplexity seat: ADJUST)
Adopted in full. Where this amendment conflicts with the text above,
the amendment governs.

## A1. Param arithmetic fix
7 layers = 704.7M params = 21.93% of 3.213B — NOT 25%. V-depth =
7 BI-ranked layers + FFN-intermediate trim sized to land the total cut
at 25% ±1%; recorded as depth+trim. ALL variants matched on TOTAL
params (tied embedding counted once); packed GGUF bytes also reported
for every variant.

## A2. Architecture diversity (7 variants)
Anchors: V-width, V-depth (7L+trim), V-KV (KV 8→4 + trim),
V-mixed (byte allocation, 16k design context — the PREDECLARED byte
candidate, §A3d). Intermediate mixtures added:
- **V-depth-lite** — 4 layers + FFN trim.
- **V-KV-lite** — KV groups 8→6 + FFN trim.
- **V-mixed-shallow** — byte allocation at 8k design context.
Any variant dropped is documented with its reason in STAGE1-REPORT.md.

## A3. Frozen analysis definitions (frozen BEFORE any timing)
(a) **Traffic formula.** B = B_W + B_KV(S), where B_W = actual tensor
    payload bytes of the variant (packed GGUF tensor bytes), and
    B_KV(S) = 2 (K&V) · 2 (bytes, fp16) · S · Σ_ℓ H_KV,ℓ · 128, with
    S = MEAN occupied context over the timed interval
    (= prompt tokens + 64 for a 128-token decode).
(b) **Comparison models.** M_P: y ~ thread + P + S + P·S.
    M_B: y ~ thread + B_W + B_KV(S). PLUS the raw single-cost
    predictor B_W + B_KV(S) evaluated as-is. NO architecture-specific
    intercepts in any model.
(c) **Objective.** PRIMARY = CPU resource time/token (user+system,
    /proc process accounting). Elapsed latency is secondary. Thread
    caps (2, 4) analyzed SEPARATELY.
(d) **Claims.** V-mixed is the predeclared byte candidate — no post-hoc
    "best variant" winner-picking. The 10% claim (funding hurdle)
    requires the upper CI bound of the paired log-time ratio
    (V-mixed vs V-width) < 0.90 at BOTH 8k and 16k. The SCIENTIFIC
    criterion is prospective predicted-vs-observed contrast agreement,
    reported three-way: PASS / practically negative / inconclusive.
    (This replaces the v0.3 PASS/FAIL (a)+(b) pair: (a) survives as
    the M_B-vs-M_P held-out comparison in §A4; (b) survives only as
    the funding hurdle above, applied to V-mixed alone.)

## A4. Measurement hygiene
- Fixed token IDs; exactly 128 decode steps replayed identically
  across variants (same prompt token IDs, temp 0, same step count).
- Cache capacity provisioned > 16k+128; no eviction/shift during a
  timed interval.
- Prediction evaluated HOLDING OUT whole architectures
  (leave-one-variant-out); main metric = held-out relative prediction
  error; Spearman ρ secondary.
- ≥10 randomized paired timing blocks per cell if runtime allows;
  otherwise the achieved precision is reported honestly.
- HF-vs-GGUF numerical output check on fixed sequences for EVERY
  variant BEFORE timing (export errors ≠ quantization effects);
  base-model HF-vs-GGUF agreement is the reference level.

# Amendment Round 2 — Gate 2 CLOSED (2026-10-23, Gemini seat: ADJUST)
Panel closed ADJUST/ADJUST (Perplexity + Gemini). Merged here; where
this conflicts with earlier text, this governs.

## B1. SIMD alignment (binding)
Every pruned dimension kernel-aligned: FFN intermediate sizes in
multiples of 256 (Q4_K_M super-block); projection widths multiples of
64; head_dim stays 128; KV-group removals keep GQA divisibility
(24 Q heads ÷ KV heads integer). After GGUF export, a kernel sanity
check per variant (llama.cpp verbose/perf output): any variant showing
fallback-path behaviour is flagged — its timing is confounded, not
evidence. Consequence for matching: keep_inter granularity is now
256, so param matching is best-effort under alignment; exact
removed fractions are reported per variant and P enters both
regression models as a covariate. Alignment beats count.

## B2. Layer-overhead control
Layer count L is a covariate in BOTH M_P and M_B; the L coefficient is
reported. Depth-matched control: V-depth-alt = 21 layers retained with
the 7 removed layers chosen by an evenly-spaced rule (not BI) + FFN
trim — isolates the BI layer-choice effect. (The literal "21 layers,
cut mostly from FFN" is arithmetically impossible: 7 layers alone are
21.9% of the 25% cut. Recorded as the reason for this form.)

## B3. Context grid
Occupied contexts: 512 / 2048 / 4096 / 6144 / 8192 / 12288 / 16384.
The prediction contest is evaluated at C ≥ 4096 ONLY (at 512 the two
cost models are collinear by construction — reported, not contested).

## B4. "Clearly beats" operationalized
Traffic model wins prediction iff, on the C≥4096 subset: held-out
(leave-one-variant-out) relative prediction error is lower for M_B
than M_P AND ΔAIC > 10 in M_B's favour. Saturation caveat: if 4-thread
16k tok/s does not exceed 2-thread, that cell is a saturation regime —
excluded from the PASS contrast, reason stated.

## B5. Measurement protocol
taskset pinning of the server to fixed cores; 5 warmup runs per cell
discarded (implemented as 5×32-token warmups after the prime — same
kernels/caches, quarter cost; deviation noted); 10 measured repeats
per cell where runtime allows (≥5 with precision reported otherwise;
cells at C<4096 run 5 by design — report-only); governor recorded
(not exposed in this VM — scaling_governor absent; recorded as such);
mean / SD / 95% CI reported per cell.



## C. Runtime amendments during stage 1 (2026-10-24, agent judgment under
the standing autonomous-gates instruction; frozen criteria in §A3/B4
UNCHANGED; all amendments made BEFORE any byte-favoured variant was
measured)
C1. REPS: B5's "else >=5" clause invoked — 5 reps/cell. Measured decode
    at 8k cap2 was 1.42 tok/s wall; 10 reps x full grid projected 40+ h
    on this 2-core shared VM.
C2. GRID: B3's "if runtime allows" clause invoked — 12288 dropped;
    grid = 512/2048/4096/6144/8192/16384 (B3 minimum: 2048+6144 kept).
C3. SERVER-PROCESS PROTOCOL (supersedes session structure): one FRESH
    llama-server process per (variant x cap x ctx) cell — prime, 5x32
    warmups, 5 reps contiguous. Reason (instrument artifact, observed
    2026-10-24): with one server process serving multiple ctxs, decode
    speed at a ctx depends on the slot's ctx-switch history — V-width
    cap2 16384 measured 1.62/1.71 CPU-s/token (server-reported decode
    155.9/171.8 s per 128 tok) in multi-ctx sessions vs 0.64/0.62
    (58.3/55.7 s) in a near-virgin session; 8192 cells were consistent
    (0.81-0.93) across processes. All pre-amendment reps are EXCLUDED
    from the analysis (kept in the JSONL, unflagged); analysis uses
    "virgin": true reps only. Block pairing uses the within-cell rep
    ordinal (0-4).
C4. PRIMES are prefill-only (n_predict=1): the prime exists to populate
    the KV cache; its 128-token decode tail pushed loaded 16k primes
    past the request timeout. Prime cpu_s is descriptive only.
C5. RESILIENCE: any request error/timeout restarts the server and
    retries the cell (3 consecutive failures -> skip; checkpoint resume
    refills). Motivated by one observed llama-server deadlock on a
    16k->2k prefix-truncation request and repeated VM restarts.

# Amendment Round 3 — Interim panel (2026-10-24, Perplexity + Gemini
seats: ADJUST / ADJUST, no KILL; record PANEL-INTERIM-2026-10-24.md).
Rolling independent reviews every 4 h ordered by the user 2026-10-24
20:13 CDT; adopted fixes are appended to REVIEW-FIXES-INBOX.md and
applied under the standing instruction. Frozen criteria in A3/B4 are
unchanged EXCEPT where D1 explicitly supersedes C3's pairing rule
for the funding contrast. All D-items were adopted BEFORE V-mixed,
V-KV, V-KV-lite, or V-depth-alt were measured (V-width, V-depth
complete; V-depth-lite partially measured under the virgin protocol —
its banked reps stand; its remaining cells join the D2 pool).

## D1. Funding contrast — genuine temporal blocks (BLOCKING)
C3's "block pairing uses the within-cell rep ordinal" is SUPERSEDED
for the funding contrast: rep ordinals from processes run at
different times are NOT paired blocks (temporal confounding on a
shared VM). The funding contrast (V-mixed vs V-width, caps 2 and 4,
ctx 8192 and 16384) is measured in GENUINE blocks: 5 blocks per
(cap, ctx); each block = one fresh-process virgin cell of V-width
and one of V-mixed, ADJACENT in time, order randomised within the
block; each block-cell = prime + 5x32 warmups + exactly 1 rep
(128 decode steps, identical replay, same CPU accounting). Records
carry "pair_block": 0..4 and "sched": "funding". The frozen hurdle
(upper 95% CI of the paired log-time ratio < 0.90 at 8k AND 16k,
per cap) is evaluated ONLY on these pair_block reps, paired by
(cap, ctx, pair_block). Existing virgin V-width reps remain valid
for the prediction analysis and as descriptive data; they are NOT
paired controls. If fewer than 5 blocks are achieved at any
(cap, ctx), the hurdle is reported NOT VALIDLY EVALUABLE there —
no unpaired analysis may be substituted and called the frozen test.

## D2. Interleaved schedule for remaining variants
All remaining unmeasured variants (V-depth-alt, V-KV, V-KV-lite,
V-mixed, plus V-depth-lite's remaining cells) are measured in a
randomised INTERLEAVED pool (one global shuffled cell order across
variants/caps/ctxs) instead of fixed per-variant sequence.
V-mixed's 8192/16384 cells are NOT in the pool: its 5 reps at those
cells come from the D1 funding blocks (1 per block). Driver:
scripts/measure_blocks.py (phases: funding -> diagnostic -> pool);
same virgin cell protocol, prime, warmups, replay, accounting.

## D3. Diagnostic remeasurement (predefined, small)
Fresh-process remeasurement, originals preserved and never replaced:
V-depth at 6144 and V-width + V-depth at 16384, both caps. Each
diagnostic cell (5 reps) is measured ADJACENT to a reference cell =
the other variant of the {V-width, V-depth} pair at the same
(ctx, cap), order randomised within the diagnostic block. Records
carry "diagnostic": true and are analysed SEPARATELY (diagnostic
cell mean vs original cell mean); they never enter the primary
fits, cell stats, or contrast. New reps in ALL phases record extra
covariates: involuntary context-switch delta of the server process
(/proc/<pid>/status nonvoluntary_ctxt_switches) and 1-min host
loadavg at rep start; primes/reps also record prompt_n and server
timings (occupied tokens, prefill activity, decode steps); affinity
is constant (taskset -c 0,1) for every measurement in this study.

## D4. V-KV-lite adjudication + teacher-forced check
Timings are KEPT (mechanism argument: degenerate tokens traverse
identical kernels/traffic in a pure timing stage) AND a
teacher-forced HF-vs-GGUF logits/distribution check runs on
identical token sequences, with the BASE model's HF-vs-GGUF
discrepancy as reference (script scripts/check_kvlite_logits.py;
the HF checkpoint is regenerated deterministically by
scripts/prune_variants.py, whose construction produced the GGUF).
PREDECLARED rule: PASS iff dNLL(V-KV-lite) <= dNLL(BASE) + 0.05
nats/token AND meanKL(V-KV-lite) <= 2 x meanKL(BASE) AND
top1-agreement(V-KV-lite) >= top1-agreement(BASE) - 0.10, where
dNLL = mean teacher-forced NLL(GGUF) - NLL(HF) over the evaluated
positions. If FAIL: V-KV-lite is excluded from the PRIMARY
prediction fit and the excluding-sensitivity becomes primary for
it (documented as a drop under the build/export provision);
timing data retained either way. If PASS: primary fit includes
it; the excluding fit is the sensitivity analysis.

## D5. Analysis definitions FROZEN (analyze.py, 2026-10-24, before
the remaining architectures' data lands)
- Row selection (executable, in analyze.py): PRIMARY reps =
  kind=rep AND "virgin" AND NOT "diagnostic". Pre-virgin reps
  (30, V-width cap2, multi-ctx sessions) are EXCLUDED; excluded
  rows, restarts, timeouts, and incomplete attempts are counted
  and reported in STAGE1-REPORT.md.
- Per cap, C>=4096 reps only. M_P: y ~ 1 + P + S + P*S + L.
  M_B: y ~ 1 + B_W + B_KV(S) + L. (P in billions, S = (ctx+64)/1e3,
  B in GB, L = layer count; y = CPU user+sys s/token, rep-level.)
  No architecture intercepts. Raw single-cost B_W + B_KV(S)
  reported separately, as-is.
- LOVO aggregation: for each held-out variant in the fit set, fit
  on the other variants' reps, predict the held-out variant's
  reps; metric = MEAN of rep-level |pred - obs| / obs over all
  held-out reps. AIC observational unit = individual rep
  (AIC = n ln(RSS/n) + 2k). Traffic wins per B4 on the PRIMARY
  fit set (D4 governs V-KV-lite's membership).
- Cell stats: mean / SD / 95% CI (t) per cell over primary reps.
  Funding uncertainty is at the block level (D1 pairing); cell
  CIs are within-process and are reported as such.
- Saturation: per B4, via measured wall tok/s (cap4 <= cap2 at
  16k => saturation regime; the funding contrast cell is flagged
  and excluded from the PASS contrast with the reason stated).

## D6. Record corrections (panel reporting items)
- C2 citation corrected: B3 contains NO literal "if runtime
  allows" clause. The 12288 drop is a RUNTIME GRID AMENDMENT made
  under agent judgment (C1's measured-rate basis), not a B3 clause
  invocation. C2's text stands as history; this note governs.
- V-mixed-shallow disposition: DROPPED before measurement.
  Marginal-byte greedy allocation produced a structurally
  identical allocation at the 8k and 16k design contexts (differed
  only in the design-S label), so it was not a distinct variant;
  the identity is itself recorded as a stage-1 observation in
  STAGE1-REPORT.md.
- Bytes: tensor-payload bytes (gguf-py sum of tensor data) and
  GGUF file bytes (results/packed_bytes.json) are reported
  SEPARATELY per variant; B_W in the traffic model is the
  tensor-payload figure (A3a).

## D7. Interpretation note (Gemini seat correction)
Gemini's claim that V-depth "does not trim KV read bytes" is
factually wrong and must not be cited: removing 7 of 28 layers
removes ~25% of KV read traffic as well (21 layers x 8 heads vs
28 x 8). Perplexity's reading stands: V-depth's flat/slower
long-context result is unfavourable descriptive evidence for a
simple bytes->time account; it is not a test of V-mixed.

## Amendment Round 4 (2026-10-25, interim panel Round 3: ADJUST/ADJUST)

## E1. Restart-split funding block replacement (protocol)
Funding block 1 at cap2/8192 is replaced IN FULL (one new adjacent
pair of fresh-process virgin cells, order randomised). The original
reps are preserved; the original block is invalid for the primary
D1 contrast. Exclusion reason: D1 temporal-adjacency violation
(pair split by the 04:35 UTC VM restart) — never the measured
value (the V-width member's 2.328 CPU-s/token is not the reason).

## E2. Block-aware restart handling + full funding audit (protocol)
Every funding block was audited against restart times. Audit
result (coordinator, 2026-10-25): three restart-split blocks —
cap2/8192 blk1 (restart ~04:35 UTC), cap2/16384 blk0 (~09:06 UTC),
cap4/16384 blk0 (~13:34 UTC) — all three replaced whole under the
E1 rule in a REPAIR phase (funding -> repair -> diagnostic ->
pool), reps tagged "repair": true; analyze.py prefers repair reps
in the D1 pairing and reports them. Going forward, funding resume
is block-aware: an incomplete block re-runs whole rather than
resuming its missing member against a pre-interruption member.
Interrupted attempts and lost work are recorded in
STAGE1-RUN-STATUS.md.

## E3. No outcome-driven reruns (protocol)
The completed cap2/16384 result stands even though it fails the
hurdle. No added blocks or re-tests except independently
documented protocol failures (E1/E2 class).

## E4. Phase order (protocol)
Funding (incl. E1/E2 repairs) -> ALL D3 diagnostics (V-depth/6144
first) -> interleaved pool. Diagnostics never enter primary fits;
a diagnostic-revealed measurement defect is repaired before the
pool starts, with affected earlier measurements documented.

## E5. Cap pattern = exploratory only (analysis)
Cap comparisons at matched contexts with cap-specific estimates
and load/context-switch sensitivities. The frozen primary hurdle
and prediction models are unchanged. cap4/8192 vs cap2/16384
confounds cap with context and is not a clean cap comparison; the
0.771 traffic prediction was for cap2/8192 only.

## E6. Thread-cap description (reporting)
Caps are llama.cpp thread counts (-t 2 / -t 4) under a taskset
-c 0,1 pin to 2 cores: "cap4" is 4 threads on 2 cores, NOT 4
cores. Saturation is evaluated with measured wall tok/s, not
CPU-s/token alone.

## E7. Hurdle / verdict separation (reporting)
Failure of the funding hurdle does NOT by itself assign the
three-way scientific verdict, which is decided by the frozen
C>=4096 LOVO comparison, dAIC, and prospective contrast agreement
on the full eligible data. Interim status language: cap2/8192 has
four valid D1 blocks until E1 lands (NOT VALIDLY EVALUABLE);
cap2/16384 is complete and fails the hurdle; cap4/8192 meets its
cell-level hurdle (upper CI 0.842). The stale 2026-10-25 02:10
UTC analysis file's zero-pair contrast is not current.
[G2 CORRECTION (Round 4 supplement, 2026-10-25): the cap2/16384
phrasing above is superseded — cap2/16384 has FOUR valid D1
blocks and is NOT VALIDLY EVALUABLE until its prescribed E2
whole-block repair lands; the raw interval is provisional
descriptive evidence only, and the repaired cell is judged on
its own 5 valid blocks. Same for cap2/8192 pending its E1
repair. E7's hurdle/verdict separation stands.]

## E8. V-KV-lite (protocol, unchanged)
D4 stands: excluded from the primary prediction fit, timings
retained, including-fit as sensitivity; timings are not evidence
for a functionally valid pruning candidate. Cells stop only under
a prospective written resource-based amendment.
