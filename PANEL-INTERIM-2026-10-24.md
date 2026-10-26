# Interim Review Panel — Stage 1 in-progress (2026-10-24)

Materials: INTERIM-REVIEW-PACKET-2026-10-24.md (snapshot 20:02 CDT:
127/420 valid virgin reps; V-width + V-depth complete; V-mixed,
V-KV, V-KV-lite, V-depth-alt unmeasured) + SPEC-v0.3.md. Both seats
blind, identical packet.

## Perplexity seat (Pro Search) — ADJUST
Thread: https://www.perplexity.ai/search/ed5bccd5-34cf-401f-9ae7-07739ff59bc5
No fatal flaw requiring stopping Stage 1 now. Two issues would
invalidate specific confirmatory claims if unresolved:

1. **Temporal confounding of the funding contrast (main catch).**
   Fixed variant order + contiguous reps on a shared VM confound
   architecture with measurement period. Rep ordinals 0-4 from a
   V-width process run days before a V-mixed process are NOT paired
   blocks. Required fix BEFORE V-mixed is measured: genuine temporal
   blocks for the funding contrast — at each cap and primary context
   (8192, 16384), V-width and V-mixed in adjacent fresh-process
   cells, order randomised within block, repeated across blocks.
   Existing virgin V-width data = descriptive/prediction data, not
   retroactively paired controls. If genuine blocks are infeasible,
   report the funding hurdle as NOT VALIDLY EVALUABLE under the
   paired design — do not substitute an unpaired analysis and call
   it the frozen test.
2. **V-KV-lite export validity unresolved.** Buffer scaling + no
   kernel fallback do not establish correct tensor mapping. Resolve
   with teacher-forced HF-vs-GGUF logits/distribution comparison on
   identical token sequences (base quantisation discrepancy as
   reference; unquantised GGUF intermediate if feasible). If
   correctness cannot be established: document and drop under the
   spec's build/export provision. Timings until then = diagnostic
   only, not primary-comparison evidence.

Further required fixes:
- [protocol] Small predefined diagnostic remeasurement of the
  conspicuous cells (V-depth 6144; both complete variants at 16384;
  both caps), fresh processes, randomised adjacent reference cells;
  preserve originals, never selectively replace. Record occupied
  tokens, cache/prefill activity, decode steps, CPU deltas, elapsed,
  affinity, host-load indicators.
- [analysis] Correct uncertainty unit: means/SDs/95% CIs (not only
  medians); reps nested within process cells / temporal blocks;
  caps separate; saturation exclusion via measured elapsed tok/s.
- [analysis] Freeze the executable prediction comparison before
  more architectures arrive: LOVO aggregation, AIC observational
  unit, final formulas (P and L in both models); raw byte predictor
  reported separately. Two complete architectures cannot yet support
  the cross-architecture prediction claim.
- [reporting] Make the pre-C3 exclusion executable/auditable
  (row-selection rule frozen; report excluded rows, restarts,
  timeouts, incomplete attempts).
- [reporting] Reconcile the record: document V-mixed-shallow's
  disposition; C2's citation of a B3 "if runtime allows" clause is
  not literally in B3 — describe 12288's removal as a runtime grid
  amendment; report tensor-payload bytes separately from GGUF file
  bytes in the traffic predictor.
- Interim conclusion allowed (its wording): V-depth shows no
  consistent long-context advantage over V-width (cap2: 4.9% slower
  @8192, 14.1% slower @16384; cap4: 1.1% slower @8192, 3.1% faster
  @16384) — unfavourable descriptive evidence for a simple
  "fewer logical bytes = less CPU time" account, but it does NOT
  establish failure of V-mixed or of the held-out traffic-model
  comparison.
- Would KILL the current run only if controlled fresh-process checks
  show identical cells still execute different workloads / retain
  uncontrolled state large enough to make the 10% contrast
  unidentifiable with no feasible correction on this VM.

## Gemini seat (Deep Research) — ADJUST
Conversation: https://gemini.google.com/app/ea576b19d2b6cc37
Fatal flaws: none. Converges with Perplexity on the main catch
(fixed sequential variant order = temporal confound on a shared VM)
and on the remedy's direction. Its required fixes:
- [protocol] Interleave ALL remaining unmeasured variants
  (V-depth-alt, V-KV, V-KV-lite, V-mixed) in randomised blocks
  rather than fixed sequence.
- [protocol] Re-run V-depth 6144 at both caps under fresh process /
  interleaved scheduling to decide host preemption vs reproducible
  kernel behaviour.
- [analysis] Record involuntary context switches
  (/usr/bin/time -v) as a diagnostic covariate for preemption.
- [reporting] Full-dataset regressions + sensitivity sub-analysis
  excluding V-KV-lite.
Splits from Perplexity on two points:
- V-KV-lite: Gemini says keeping it in timing is sound — a pure
  mechanism test; degenerate tokens traverse identical kernels and
  generate the architecture's true memory traffic.
- V-depth interpretation: Gemini says V-depth does NOT damage the
  byte hypothesis because it "does not trim KV cache read bytes."
  IN-HOUSE CORRECTION: that premise is wrong — removing 7 of 28
  layers removes ~25% of KV read traffic too (21 layers x 8 heads
  vs 28 x 8). Perplexity's reading stands: V-depth's flat/slower
  long-context result IS unfavourable descriptive evidence for a
  simple bytes->time account, though still not a test of V-mixed.
  (Also minor: Gemini's table mislabels file bytes as tensor bytes.)

## Panel synthesis (in-house) — ADJUST / ADJUST, no KILL/reject
Agreed and adopted: (1) temporal blocking/interleaving before and
during the remaining variants, with genuine adjacent V-width blocks
at the funding contexts so the V-mixed contrast is honestly paired;
(2) predefined diagnostic remeasurement of V-depth 6144 and the
16384 cells, originals preserved; (3) context-switch covariate;
(4) analysis/reporting fixes from both seats (nested uncertainty,
means/SD/CI, frozen LOVO/AIC definitions, auditable virgin-row
selection, record reconciliation).
V-KV-lite adjudication: timings KEPT (Gemini's mechanism argument),
AND Perplexity's teacher-forced logits check runs; if the check
fails, V-KV-lite is excluded from the primary prediction fit and
the sensitivity analysis becomes the primary one for it. Data kept,
claim protected.
Gemini's fixes + this adjudication forwarded to the coordinator
2026-10-24 ~20:14 CDT. Collection continues uninterrupted.

## In-house disposition
Perplexity fixes forwarded to the pilot coordinator 2026-10-24
~20:12 CDT, before V-mixed collection (V-mixed is last in the
sequence), per the standing auto-fix rule. Collection continues.

---

# Round 3 — interim panel (2026-10-25, snapshot 08:20 CDT / 13:20 UTC)

NOTE: a Round 2 packet (snapshot 2026-10-25 04:14 CDT) was written
but never panelled — no panel entry, no watermark advance. Its
questions were carried forward into the Round 3 packet
(INTERIM-REVIEW-PACKET-2026-10-25-R3.md) and answered here. Round
numbering skips the unpanelled round rather than overwriting it.
Materials: Round 3 packet + SPEC-v0.3.md. Both seats blind,
identical packet (Perplexity received the full packet text plus a
condensed statement of the frozen spec criteria, not the full spec
verbatim, because its upload UI produced no file input; Gemini
received both files in the successful regular-mode chat).

Data position at the snapshot (verified against
results/stage1_measurements.jsonl): 268 rows; reps 189 = pool 129 +
funding 30 + pre-virgin excluded 30; diagnostics 0. Funding 30/40:
cap2/8192, cap2/16384, cap4/8192 complete (5 blocks each);
cap4/16384 not yet banked. Pool unchanged at 129 (25 complete cells).
No STAGE1-REPORT.md; no stage transition.

Funding contrasts computed for the packet (t-based 95% CI on paired
log ratios, verified in-house against the JSONL):
- cap2/8192: mean log −0.326, exp CI [0.358, 1.457]; excl. block 1:
  [0.797, 1.079].
- cap2/16384: mean log −0.011, exp CI [0.939, 1.041] (all five
  ratios 0.93–1.02).
- cap4/8192: mean log −0.200, exp CI [0.795, 0.842] (all five
  ratios 0.79–0.83) — meets the cell-level hurdle on its own.
- cap4/16384: missing at snapshot.

## Perplexity seat (Search mode, NOT Deep Research) — ADJUST
Thread: https://www.perplexity.ai/search/a1ba961b-c839-454f-9e4b-24b5695b452f
Upload failed (no file-input element appeared); packet + spec
summary pasted instead. Fatal flaws: none.
Key fixes (8, smallest first):
1. [reporting] cap2/8192 has FOUR valid D1 blocks, not five;
   its hurdle is NOT VALIDLY EVALUABLE until replacement.
   cap2/16384 (CI [0.939, 1.041]) does not meet the hurdle.
   Do not quote the stale analysis file's zero-pair result.
   Correction adopted: under the frozen plan the completed
   cap2/16384 result already prevents cap2 meeting the
   two-context hurdle; the cap2/8192 repair cannot change that.
2. [protocol] Replace the ENTIRE split block at cap2/8192
   (both members, fresh adjacent virgin pair, randomised order)
   before funding closes; preserve originals, mark them invalid
   for the primary D1 contrast, link the replacement.
3. [protocol] Block-aware restart handling + audit of EVERY
   funding block against restart times/process/order; any
   restart-straddling pair is invalidated and replaced; after
   interruption, restart the whole incomplete block, not just
   its missing member.
4. [protocol] No outcome-driven cap2 rerun: retain completed
   cap2/16384; a rerun needs independently documented protocol
   failure, not hurdle failure. Funding-hurdle failure does NOT
   by itself assign the three-way scientific verdict.
5. [protocol] Keep funding → diagnostics → pool; D3 completes
   before pool expansion (V-depth/6144 first); if diagnostics
   reveal an active defect, repair before the pool.
6. [analysis] Cap pattern = interaction hypothesis, tested at
   matched contexts only; cap4/8192 vs cap2/16384 confounds cap
   with context. The 0.771 prediction was for cap2/8192; its
   proximity to the cap4 result does not validate it.
7. [analysis] Scientific assessment separate from funding;
   saturation via tok/s; audit affinity vs taskset -c 0,1 and
   nominal caps 2/4 and disclose any discrepancy.
8. [protocol] Keep D4 V-KV-lite handling (excluded from primary
   fit, timings retained, including-fit sensitivity); stop its
   remaining cells only under a prospective resource amendment.
Interim conclusion allowed (its wording): V-mixed meets the
funding threshold at cap4/8192, whereas the completed cap2/16384
cell does not, so cap2 cannot meet its two-context hurdle under
the frozen collection plan; cap2/8192 additionally requires
replacement of one invalid split block. The observed advantage
varies across measurement conditions, but cap4/16384,
diagnostics, and the remaining pool data are needed to
distinguish cap/context effects from execution conditions and
assess the separate scientific criterion.
Would change to PURSUE: audit establishes five genuinely
adjacent valid blocks per cell after replacements, restart-aware
scheduling operating, D3 reveals no unresolved defect. To KILL:
valid pairing/provenance cannot be reconstructed or maintained,
or diagnostics identify an uncorrectable systematic defect.

## Gemini seat (regular mode, Gemini Auto) — ADJUST
Chat: https://gemini.google.com/app/d4af8dfa7e93cd1f
Deep Research tried first and failed twice with Gemini system
errors; the first chat also could not read the uploaded files.
The successful review ran in a new regular-mode chat where both
files were read. (Session note: the Gemini page twice loaded
signed-out; the login specialist restored the same AI_Peppers
account both times, no account switched or signed out.)
Fatal flaws: none. Fixes (3):
1. [protocol] Invalidate funding block 1 at cap2/8192 and re-run
   as a genuine adjacent virgin pair per D1.
2. [protocol] Restart-aware block validation rule: any funding
   pair split across a VM restart is automatically flagged and
   re-run before funding concludes.
3. [reporting] Audit all funding blocks for restart-straddling;
   report dropped cells, restart counts, timeout events in
   STAGE1-REPORT.md per D5.
Interim conclusion allowed (its wording): V-mixed meets the
funding hurdle at cap4/8192 with a consistent ~18% speed
advantage, but shows no advantage at cap2/16384, where the
confidence interval centers near no effect. Because cap4/16384
remains unmeasured and cap2 funding cells fail the hurdle, the
overall funding outcome remains contingent on completing cap4
data and resolving block validity.
Would change verdict: cap4/16384 completion showing whether the
cap4 advantage extends across long contexts; or unmitigated
systemic VM degradation causing widespread unrecoverable block
corruption.

## Panel synthesis (in-house) — ADJUST / ADJUST, no KILL/reject
Seats CONVERGE on: (1) block 1 at cap2/8192 is invalid and must
be replaced whole before funding closes; (2) a restart-aware
block rule + full funding-block audit before funding closes;
(3) funding → diagnostics → pool ordering stands; (4) D4
V-KV-lite handling stands. No seat used KILL or reject-type
language; no alert fires.
IN-HOUSE VERIFICATION of load-bearing numbers: all three cells'
paired CIs recomputed from the JSONL and match the packet and
both seats (cap2/8192 [0.358, 1.457]; excl. block 1 [0.797,
1.079]; cap2/16384 [0.939, 1.041]; cap4/8192 [0.795, 0.842]).
No reviewer numeric error caught.
IN-HOUSE CORRECTIONS / ADJUDICATIONS:
- Gemini's stated reason for invalidating block 1 (contention
  inflation / outlier value) is MISFRAMED: the exclusion reason
  is the documented D1 adjacency violation, not the unfavourable
  timing — Perplexity states this correctly and it governs.
  Excluding a block for its value would be outcome-driven; the
  remedy (whole-block replacement) is the same, the reason is
  corrected.
- Perplexity's affinity audit (fix 7) VERIFIED in-house:
  scripts/measure_blocks.py imports start_server from
  scripts/measure.py, which launches every server under
  taskset -c 0,1. So "cap4" = 4 llama threads pinned to 2 cores,
  not 4 cores. This does NOT change the frozen hurdle (defined
  on the caps as run), but the final report must describe caps
  as thread counts under a 2-core pin and must not attribute
  the cap4/8192 advantage to more cores. Adopted as a reporting
  fix (E-fix below).
- Perplexity's precision correction adopted: cap2's hurdle
  failure is already settled by completed cap2/16384 regardless
  of the cap2/8192 repair; the packet's looser phrasing is
  superseded.
- Seats' answering depth differs (Perplexity 8 fixes, Gemini 3)
  but they agree on every shared point; no substantive
  disagreement to adjudicate beyond the block-1 reason above.
ADOPTED FIXES (forwarded to the pilot coordinator in
REVIEW-FIXES-INBOX.md, 2026-10-25): E1 whole-block replacement
of cap2/8192 block 1; E2 restart-aware block rule + full audit
before funding closes; E3 no outcome-driven cap2 reruns;
E4 D3 before pool, V-depth/6144 first; E5 cap pattern as
exploratory matched-context interaction only; E6 caps described
as thread counts under taskset -c 0,1 pin, never as core counts;
E7 funding-hurdle vs scientific-verdict separation stated in
the report; E8 V-KV-lite D4 handling unchanged.
Collection continues uninterrupted. Next rolling review (Round
4) reviews only material newer than the 08:20 CDT snapshot.

## Post-panel note (NOT part of the Round 3 review)
Checked 09:47 CDT after both seats landed: funding cap4/16384
block 0 has since banked (V-mixed 0.5363, V-width 1.4005 CPU-s/
token at load 8.8, nivcsw 54,235 — heavily contended), and the
driver pid changed with VM uptime indicating a restart at
~13:35 UTC. Whether cap4/16384 block 0 straddles that restart
is a Round 4 audit item under adopted fix E2; it does not alter
the Round 3 verdicts, which rest on the 08:20 CDT snapshot.

---

# Round 4 — interim panel (2026-10-25, snapshot 12:14 CDT / 17:14 UTC)

Materials: INTERIM-REVIEW-PACKET-2026-10-25-R4.md + SPEC-v0.3.md.
SPEC-v0.3.md was NOT edited in this round. No STAGE1-REPORT.md;
no stage transition.

Data position at the snapshot (verified against
results/stage1_measurements.jsonl): 284 rows; reps 197 = pool 129
(UNCHANGED) + funding 38 (was 30) + pre-virgin excluded 30;
diagnostics 0; repair-tagged rows 0. Pool unchanged (25 complete
cells). The raw funding count OVERSTATES valid blocks (see below).

New material since Round 3: cap4/16384 funding blocks 0–2 banked
(8 reps + 8 primes). Validity, from file order + the E2 audit:
- Block 0 INVALID, repair pending: V-mixed 0.5363 pre-restart #6,
  V-width 1.4005 post-restart (load 8.8, nivcsw 54,235) — D1
  adjacency violation, not a value judgment.
- Block 1 VALID fresh pair: V-mixed 0.7676 / V-width 1.4323,
  ratio 0.536, log −0.624. Superseded member: V-mixed 0.8018
  (pre-patch-interruption), identifiable only by file order.
- Block 2 VALID fresh pair: V-mixed 0.5723 / V-width 0.6162,
  ratio 0.929, log −0.074. Superseded member: V-mixed 0.7605
  (pre-restart #7), identifiable only by file order.
- Blocks 3–4: not yet measured at snapshot.
Valid-pair summary, cap4/16384 (n = 2): mean log ratio −0.349,
exp 0.706, t-based 95% CI exp [0.021, 23.208] — uninformative.
Hurdle NOT VALIDLY EVALUABLE at cap4/16384 (needs 5 valid blocks).
The naive all-blocks mean (0.642, CI [0.146, 2.829]) is
methodologically wrong (includes the invalid block and
first-members) and must not be quoted.
Validity-adjusted funding account: cap2/8192 4 valid blocks
(blk1 invalid, repair pending); cap2/16384 4 valid (blk0 invalid,
repair pending); cap4/8192 5 valid; cap4/16384 2 valid. NO cap
has 5 valid blocks at both contexts yet, so no funding hurdle
can pass or fail yet.
Incidents: VM restart #7 (~16:29 UTC) killed the driver mid-cell
(cap4/16384 blk2 first member banked); relaunched 16:30 UTC
(pid 1800); block-aware resume re-ran blk2 whole. Driver alive,
single instance at snapshot, next cell in flight. No new server
wedge. Disk 16 GB free (85% used); no disk incident.
ANALYSIS-CODE GAP (new, verified by reading
scripts/analyze.py): the funding pairing builds each variant's
block dict in file order, so a later non-repair rep silently
overwrites an earlier one (last-wins — which happens to select
the fresh members for cap4/16384 blocks 1–2), and it counts ANY
block present in both variants toward n_pairs, including the
three known-invalid blocks, until their repair reps land. Run
today it would report cap2/8192 as evaluable (n_pairs = 5,
including invalid blk1) and cap4/16384 with n_pairs = 3
(including invalid blk0). The frozen analysis therefore
overstates valid-block counts in the interim.

## External seats — BLOCKED on approval cards, NO VERDICTS RETURNED
- Perplexity Pro seat: browser task launched, then stalled on a
  file-upload approval card (approval_id
  connector-8a7b82a7-2df6-4443-9418-68ea85c43f93, requested
  17:22 UTC). A paste-instead-of-upload redirect was sent; the
  agent remained gated on the same card ~50 minutes later.
- Gemini seat (dedicated aipeppers01 account): same stall on a
  file-upload approval card (approval_id
  connector-6b6d4c30-44fa-4acd-ad05-ca194d60c468, requested
  17:22 UTC); paste redirect likewise did not clear the gate.
- Neither stall is a quota/limit message and neither session was
  signed out. Per the blocked-seat rule both seats are recorded
  as OWED at Round 5 (re-launch fresh; do not retry these
  stalled windows — they were stood down at round close).
- NO seat returned any verdict, so NO seat used KILL or
  reject-type language; no alert fires. The round is completed
  on the in-house check below, per the body's fallback rule.

## In-house check — ADJUST, no fatal flaws
Fatal flaws: none. In-house verification: every packet number
recomputed from the JSONL (cap4/16384 valid pairs, n = 2 CI,
superseded members, raw-vs-valid block counts) and the
analyze.py pairing behaviour read directly from the code;
no numeric error in the packet caught.
Adjudications / adopted fixes (forwarded to the pilot
coordinator in REVIEW-FIXES-INBOX.md, 2026-10-25, F-series):
- F1 [analysis]: exclude the three named invalid blocks
  (cap2/8192 blk1, cap2/16384 blk0, cap4/16384 blk0) from
  analyze.py n_pairs until their repair reps land; document
  the last-wins supersession rule explicitly in the analysis
  notes; do NOT run or quote any analyze.py contrast until
  the exclusion is applied and hand-verified against the
  JSONL. This is a recorded analysis amendment, not a SPEC
  edit (SPEC-v0.3.md untouched this round).
- F2 [protocol]: no change to cap4/16384 blocks 3–4 procedure;
  complete 5 valid blocks before any cap4/16384 claim; NO
  load/nivcsw covariate adjustment — the frozen analysis has
  none, and adding one mid-collection would break it.
- F3 [protocol]: funding → repair → diagnostic → pool order
  stands; the cap4/16384 instability STRENGTHENS the case for
  diagnostics before the pool (diagnostics test whether the
  instability is a measurement defect).
- F4 [reporting]: interim language fixed — cap4/16384 has 2
  valid blocks, CI [0.021, 23.208], NOT VALIDLY EVALUABLE;
  the naive 0.642 figure is never quoted.
- F5 [protocol, unchanged]: V-KV-lite D4 handling stands
  (excluded from primary fit, timings retained).
Interim conclusion allowed (in-house wording): V-mixed meets
the funding hurdle at cap4/8192 alone, but no cap has five
valid blocks at both contexts yet, so no funding hurdle can
pass or fail; cap4/16384's two valid blocks are too noisy to
support any interim effect claim.
Would change the in-house verdict to KILL: proof that valid
pairing/provenance cannot be reconstructed, or diagnostics
identifying an uncorrectable systematic defect. To PURSUE:
5 valid blocks per funding cell with repairs landed and the
F1 exclusion verified.
Collection continues uninterrupted. Next rolling review
(Round 5) reviews only material newer than the 12:14 CDT
snapshot, PLUS the two external seats owed from this round.

## Late addendum to Round 4 — Gemini seat signed out (no verdict)
After round close, the revived Gemini browser task reported it had
stopped without submitting: its original tab closed unexpectedly
during the change to pasting the packet text, and on reopening
Gemini showed a signed-out landing page with no AI_Peppers
account in the sidebar. No packet text was pasted, no verdict
was obtained, and no quota/limit message appeared — the blocker
is solely the signed-out session. Per the seat's stop rule
(stop on sign-out, no retries, no account switching), the
Gemini seat was NOT re-signed-in from this run and remains OWED
at Round 5, alongside the earlier approval-card block. The
dedicated-account vault re-sign-in standing rule was not invoked
here because the seat's own instruction prescribes stopping;
whether to restore the aipeppers01 session before Round 5 is a
call for the main agent, not this run.

## Round 4 supplement — Perplexity seat (LATE, delivered after round close)
Thread: https://www.perplexity.ai/search/92860c52-d082-4389-a618-c1ef02402e2c
Mode: Deep Research (per the seat's report; the harness-measured
browser-task window was 12:52–1:31 PM CDT, while the seat reports
its Perplexity thread itself ran 1m 19s — the two figures measure
different spans and are not in conflict). Signed in as Ammu Dabbu
throughout; no sign-out, account switch, session clearing, or
quota/limit message. Prompt pasted directly (no file upload
retried, per the change of plan). The seat reports two accidental
test markers inserted during prompt construction were removed
before submission; that process detail is unverifiable from here
and has no bearing on the verdict.
Materials reviewed: the Round 4 packet (12:14 CDT snapshot) +
SPEC-v0.3.md, as pasted by the parent after the seat's upload
route was abandoned. The verdict below rests ONLY on that
snapshot, not on later data.

## Perplexity seat (Deep Research) — ADJUST
Fatal flaws: none. Required fixes (smallest first):
1. [reporting] Correct interim status language: E7's "cap2/16384
   is complete and fails hurdle" conflicts with E2's invalidation
   of its block 0. Replace with "four valid blocks; hurdle not
   validly evaluable until the prescribed whole-block repair,"
   retaining the raw result as explicitly provisional descriptive
   evidence.
2. [analysis] Correct funding eligibility and eliminate silent
   duplicate resolution. Exclude the three named invalid original
   blocks, identify the two superseded singleton members
   explicitly, and validate counts. Record the amendment BEFORE
   patching analyze.py; do not wait for the final report.
3. [protocol] Make attempt provenance explicit. Preserve the raw
   JSONL, add an immutable row-to-attempt/status manifest for
   existing records, and bank attempt identifiers for subsequent
   blocks. A complete replacement pair — not an individually newer
   member — must be the unit of selection.
4. [protocol] Complete the existing schedule without
   outcome-driven additions: cap4/16384 blocks 3–4, the three
   prescribed whole-block repairs, then all D3 diagnostics before
   the interleaved pool. Resolve any diagnostic-established
   measurement defect before pool collection proceeds.
5. [analysis] Run the unchanged frozen analyses on audited data.
   Keep funding-pair eligibility separate from D5 prediction-fit
   eligibility; no primary load/context-switch adjustment,
   selected outlier deletion, or extra blocks based on observed
   ratios. Apply the wall-tok/s saturation rule; report achieved
   precision and diagnostic results.
Answers to the packet's 7 questions (condensed; full text in the
seat's report): (1) a recorded amendment is required — last-wins
happens to recover the intended pairs but does not establish
same-attempt membership; hand-adjusted reporting does not fix
executable analysis; proposed amendment F1 (attempt-eligibility
and supersession bookkeeping, versioned manifest by immutable
row identifiers + file hash, valid block = exactly one V-width
and one V-mixed rep from the same uninterrupted adjacent
fresh-process attempt, silent last-wins prohibited) plus F2
(E7 interim-status clarification above); analyze.py may be
patched mid-collection AFTER the amendment is recorded, with
prior code/output preserved and the amended version identified.
(2) cap4/16384: (a) underpowered so far — complete 5 valid
blocks and let the CI speak — with (c)'s reporting obligation
(prominent instability warning, never labelled a demonstrated
cap4×16k architectural effect); no new load threshold, quieter
replacement runs, primary-model adjustment, or extra blocks.
(3) packet accounting confirmed: cap2/8192 4/5, cap2/16384 4/5,
cap4/8192 5/5 (cell hurdle met, upper CI 0.842), cap4/16384 2/5;
15 of 20 valid pairs; raw 38 reps = 30 valid-pair members +
6 invalid-block members + 2 superseded singletons; completion
needs 10 new reps across 5 outstanding pairs (at snapshot).
(4) no fatal flaw in any of the four developments; continuing to
count invalid pairs after identifying the defect, or selecting
replacement runs by outcome, would be a materially different
integrity problem. (5) keep funding → repair → diagnostic →
pool; V-depth/6144 first; diagnostics never enter primary fits.
(6) V-KV-lite D4/E8 unchanged (fails all three checks as
computed: dNLL 3.761 vs max 0.0402, KL 18.79 vs 0.384, top-1
0.000 vs min 0.858); stopping early would need a separate
prospective resource amendment. (7) whole-block relaunch
adequate in principle; provenance implementation must be
strengthened (durable attempt IDs, whole-attempt eligibility
checks, documented interruptions; D3 adjacent pairing preserved
after interruptions).
Interim conclusion allowed (its wording): "V-mixed meets the
cap4/8192 cell-level funding hurdle, but neither cap's
two-context hurdle is currently validly evaluable because
required valid blocks and repairs remain outstanding. The
cap4/16384 observations show substantial instability, not a
reliable effect estimate, and do not establish superiority of
the byte model or determine the final Stage-1 scientific
verdict."
Would change verdict: to PURSUE — recorded amendment +
reproducible audit recovering exactly 4/4/5/2, unambiguous
provenance on subsequent attempts, diagnostics showing no
unresolved defect before the pool. To KILL — audit cannot
reconstruct valid same-attempt membership and compliant
replacements cannot be obtained, or diagnostics establish an
unrepairable systematic defect. A valid funding-hurdle failure
alone would not trigger KILL.

## Panel synthesis, Round 4 complete (in-house ADJUST + Perplexity ADJUST; Gemini still owed)
No seat used KILL or reject-type language; no alert fires.
IN-HOUSE VERIFICATION of Perplexity's load-bearing numbers
against the JSONL at the 12:14 CDT snapshot: valid-block counts
4/4/5/2 and 15 of 20 valid pairs confirmed (4+4+5+2 = 15 pairs =
30 members); raw 38 reps = 30 + 6 + 2 confirmed; completion
need of 10 reps across 5 outstanding pairs confirmed; V-KV-lite
thresholds confirmed from the D4 rule (dNLL max −0.0098+0.05 =
0.0402; KL max 2×0.192 = 0.384; top-1 min 0.958−0.10 = 0.858);
cap4/16384 ratios and CI [0.021, 23.208] confirmed. No reviewer
numeric error caught.
IN-HOUSE ADJUDICATIONS:
- Perplexity's E7 language correction is ADOPTED, with a
  precision note: Round 3's "cap2 hurdle failure is settled by
  completed cap2/16384" is SUPERSEDED as a statement of D1
  eligibility. The raw complete-cell interval ([0.939, 1.041])
  remains provisional descriptive evidence, but the cell's
  current D1 eligibility is 4 valid blocks, and the repaired
  cell must be judged on its own 5 valid blocks — its outcome
  must not be presumed to rescue or to preserve the raw result.
  What stands unchanged: E3 (no outcome-driven reruns) and the
  requirement that cap2 needs its prescribed repairs; the repair
  itself is a prescribed protocol remedy, not a rescue re-run.
- Perplexity's F1 attempt-manifest amendment STRENGTHENS the
  in-house F1: the manifest (immutable row identifiers + file
  hash, attempt/validity/supersession status) and durable
  attempt IDs on subsequent records are adopted; whole-attempt
  selection replaces silent last-wins. analyze.py patching is
  authorized only after this amendment is recorded here, with
  prior code/output preserved and the amended version
  identified — which this record does.
- Perplexity's fixes 3–5 converge with in-house F2/F3/F4: no
  procedure change for remaining blocks, no covariate
  adjustment, diagnostics before pool, frozen analyses on
  audited data, funding eligibility kept separate from D5 fit
  eligibility. No substantive disagreement between the
  Perplexity seat and the in-house check.
ADOPTED FIXES (forwarded to the pilot coordinator in
REVIEW-FIXES-INBOX.md, G-series): G1 attempt-eligibility and
supersession bookkeeping (manifest + attempt IDs +
whole-attempt selection; analyze.py patch authorized post-
amendment with version preserved); G2 E7 interim-status
language correction for cap2/16384 (and cap2/8192) to "4 valid
blocks, not validly evaluable pending repair"; G3–G5 reaffirm
F2/F3/F4 (complete the schedule as prescribed; diagnostics
before pool; fixed cap4/16384 interim language; V-KV-lite D4
unchanged is carried in G5).
Collection continues uninterrupted. Gemini seat remains OWED
(signed out; session must be restored before it can run).

## Post-snapshot note (NOT part of the Round 4 review)
After the 12:14 CDT snapshot, cap4/16384 block 3 completed as a
valid pair: V-width 0.6238 (load 4.2) + V-mixed 0.5516, ratio
0.884, log −0.123 — in line with block 2's V-width 0.6162, not
with blocks 0–1's 1.40/1.43. VM restart #8 (~18:08 UTC) then
killed the driver (relaunch pid 1801, status file updated
18:08 UTC); block 4's first prime attempt was lost to it, and
its V-mixed rep 0.5748 banked post-restart, with the V-width
member in flight at the 18:32 UTC check (291 rows total).
Valid blocks are now cap4/16384 = 3 (blocks 1–3); outstanding
pairs are now 4 (cap2/8192 blk1 repair, cap2/16384 blk0 repair,
cap4/16384 blk0 repair, cap4/16384 blk4), needing 8 new reps.
Round 5 reviews all of this as new material; none of it alters
the Round 4 verdicts, which rest on the 12:14 CDT snapshot.

---

# Round 5 — interim panel (2026-10-25, snapshot 16:14 CDT / 21:14 UTC)

Materials: INTERIM-REVIEW-PACKET-2026-10-25-R5.md + SPEC-v0.3.md
(condensed frozen-criteria section in the packet; full spec
governs). SPEC-v0.3.md was NOT edited in this round. No
STAGE1-REPORT.md; no stage transition.

Data position at the snapshot (verified against
results/stage1_measurements.jsonl, 308 rows, md5
6891fcc12e7f86029d926713009f0149): pool 129 (UNCHANGED, 25
cells) + funding 49 reps + pre-virgin excluded 30; diagnostics
0. Retained valid D1 pairs 19/20 (was 15/20): cap2/8192 5/5
(ratios 0.977, 0.876 repair2, 0.899, 1.022, 0.824; geometric
mean 0.917, 95% CI [0.824, 1.020]); cap2/16384 5/5 (ratios
0.955 repair1, 1.013, 1.024, 1.015, 0.928; mean 0.986, CI
[0.934, 1.041]); cap4/8192 5/5 unchanged (mean 0.818, CI
[0.795, 0.842], cell hurdle met); cap4/16384 4/5 (ratios
0.536, 0.929, 0.884, 0.922; mean 0.798, n=4 CI [0.522,
1.219], NOT VALIDLY EVALUABLE). New since Round 4:
cap2/8192 blk1 repair2 and cap2/16384 blk0 repair1 pairs
landed; cap4/16384 blk4 completed (0.5763/0.6249) under
split attempt tags a2/a3 (bookkeeping bug, id computed
inside the per-variant loop, since fixed to once per
block; manifest merges them under a2 with the merge
recorded; earlier singleton 0.5748 superseded); VM
restarts #9 (~19:38 UTC, split cap2/8192 blk1 repair1)
and #10 (~21:03 UTC, split cap4/16384 blk0 repair1:
V-mixed 0.5533 banked, V-width lost); driver relaunched
21:04 UTC (pid 1763), cap4/16384 blk0 repair2 in flight
— the ONE outstanding pair. Manifest stale: built at
JSONL md5 71221415…, "pending 0" predates the cap4 blk0
repair1 singleton; rebuild instructed after the repair
phase and at MEASURE DONE before analyze.py runs; no
analyze.py contrast run or quoted this round.

## Perplexity seat (Search mode, NOT Deep Research) — ADJUST
Thread: https://www.perplexity.ai/search/5fc31ce2-893d-48b4-8929-97bfa09f4fb1
Seat history this round: first attempt (full packet +
spec as granted files) failed before submission — the
browser agent reported granted files are not readable
as pasteable text and upload was forbidden by the
assignment; second attempt (full inline text) errored
on a truncated model response with no verdict; third
attempt (condensed packet, inline) delivered. Mode
honesty note: Deep Research was attempted and opened
an upgrade dialog requiring Perplexity Max, so the
thread ran in Search mode. Signed in as Ammu Dabbu
throughout; no sign-out, switch, or upload.
Fatal flaws: none established from the packet.
Acceptance of the split-tag merge conditional on
attempt-level provenance.
Required fixes / answers (condensed; full text in the
seat's report): (1) cap2's funding hurdle is settled-
failed (5 valid blocks each context, upper CIs 1.020
and 1.041), with nothing in the rules forbidding that
while the E7 scientific verdict stays open; no
outcome-driven reruns. (2) blk4 split-tag merge
acceptable ONLY if both members are truly the same
uninterrupted adjacent fresh-process attempt and the
tag split was purely bookkeeping — preserve original
tags, driver-log evidence, explicit merge mapping; a
coordinator assertion alone is insufficient, and any
restart or unresolved attempt boundary requires a
whole-block rerun. (3) Rebuild and validate the
manifest before analyze.py: assert 20 retained valid
pairs (5 per cell), whole-attempt rule, no reused
observations, traceable-but-excluded split/incomplete/
superseded records, explicit blk4 merge and blk0
repair2 selection, no funding repair pending, counts
and intervals recomputed from exactly the retained
records; "pending 0" is currently false. The seat
notes its quoted counts/intervals are packet-reported
and not independently auditable from the packet alone.
(4) blk0 repair2 as a whole block is correct E2
handling; never combine its banked singleton with
repair2; require evidence repair2 itself completed
uninterrupted and adjacently, plus explicit
supersession of prior incomplete attempts. (5)
Funding → diagnostics → pool may proceed once
repair2 lands and validation passes; diagnostics must
specifically investigate the V-KV-lite teacher-forced
mismatch (see in-house adjudication correcting this
point below). (6) Keep the interim claim narrow; no
traffic-win claim and no three-way scientific verdict
yet. (7) PURSUE would need the frozen C≥4096 LOVO
comparison, ΔAIC > 10 for traffic, and prospective
contrast agreement with valid provenance; KILL would
need the frozen practically-negative criteria met or
an irreparable validity defect — not cap2's hurdle
failure alone; the seat notes the packet did not
supply complete practically-negative criteria.
Interim conclusion allowed (its wording): "The
strongest interim conclusion is that cap2 fails its
funding hurdle on the reported five valid blocks at
each context: geometric mean ratios are 0.917 with
95% CI [0.824, 1.020] at 8192 and 0.986 with [0.934,
1.041] at 16384. Cap4/8192 meets its cell hurdle at
0.818 [0.795, 0.842], but cap4's two-context hurdle
remains not validly evaluable because cap4/16384 has
only four valid blocks; its reported 0.798 [0.522,
1.219] supports no hurdle determination. There are
reportedly 19 of 20 retained pairs, diagnostics are
0/8, and no prediction-fit or E7 scientific verdict
is available. These counts, ratios, intervals, and
provenance assertions cannot be independently
verified from this packet alone."

## Gemini seat — BLOCKED, NO VERDICT, OWED at Round 6
The first Gemini attempt (granted files + paste
instruction) was stood down before it could hit the
same granted-file wall as Perplexity's first attempt.
The corrected attempt (full inline packet text, new
chat, dedicated aipeppers01 account) stopped at
sign-in: the session started signed out; after the
saved login was submitted, Google required a phone-
only QR-code verification ("scan the QR code with
the account holder's phone") and "Try another way"
offered no alternative. That is a challenge only the
account holder can clear, so per the seat's stop rule
the seat was NOT re-signed-in from this run, no
account was switched, no packet was pasted, and no
verdict exists. The browser tab remains on the Google
QR-code verification page for aipeppers01@gmail.com.
Gemini remains OWED at Round 6; its session must be
restored by the account holder (phone QR step) before
it can run, or it will block again. This is NOT a
quota/limit message and NOT a sign-out caused by
this run.

## In-house check — ADJUST, no fatal flaws
IN-HOUSE VERIFICATION: every packet number recomputed
from the JSONL at the snapshot and matches the packet
and the Perplexity seat (all four cells' ratios,
geometric means, and CIs above; 19/20 retained pairs;
raw 49 funding reps). No reviewer numeric error
caught. The seat's "cannot verify from packet alone"
caveat is correct for a packet-only reviewer but is
superseded in-house: the numbers ARE verified against
the track's own files, so the narrow interim
conclusion stands as verified, not merely reported.
IN-HOUSE ADJUDICATIONS:
- Perplexity fix 5 is PARTLY MISDIRECTED and is
  corrected here: D3 diagnostics do NOT investigate
  V-KV-lite. Its teacher-forced mismatch is already
  established (D4 check FAILED) and dispositioned
  (excluded from the primary fit, timings retained).
  Redirecting diagnostics to V-KV-lite would break
  the D3 protocol. Diagnostics stand as specified
  (V-depth/6144 first, then V-width+V-depth/16384,
  both caps); their standing purpose per F3 is testing
  whether the cap4/16384 instability is a measurement
  defect. V-KV-lite handling unchanged (D4/E8/H5).
- Perplexity fix 7's "practically-negative criteria
  not supplied" is a CONDENSED-packet artifact, not a
  protocol gap: the seat reviewed the condensed
  packet, which summarised the three-way verdict rule
  without the full SPEC text. No action: the frozen
  verdict rule (LOVO + ΔAIC + prospective contrast
  agreement) is already recorded in SPEC §A3d/B4/E7,
  and the scientific verdict is decided on the full
  eligible data at analysis time, not from this packet.
- Perplexity's Search-mode label is accepted as
  reported (Deep Research → Perplexity Max upgrade
  dialog). Earlier rounds' Deep Research seats are
  unaffected; the mode actually used is what is
  recorded here.
ADOPTED FIXES (forwarded to the pilot coordinator in
REVIEW-FIXES-INBOX.md, 2026-10-25, H-series): H1 cap2
hurdle settled-failed language; H2 blk4 merge
conditional acceptance with preserved evidence; H3
post-repair manifest rebuild assertions; H4 blk0
repair2 whole-block completion rule; H5 D3/V-KV-lite
adjudication above (diagnostics NOT redirected).
Collection continues uninterrupted. Gemini seat
remains OWED (holder-only phone QR step outstanding).

## Post-snapshot note (NOT part of the Round 5 review)
After the 16:14 CDT snapshot, cap4/16384 blk0 repair2's
V-mixed member banked (0.5536, attempt
f4-16384-b0-repair2); its V-width member was in flight
at the 16:40 CDT check (310 rows total). This does not
alter the Round 5 verdicts, which rest on the snapshot.


---

## Literature-base + tweaks panel — 2026-10-26 (API seats only; DeepSeek skipped per standing decision)

Materials: LIT-REVIEW-PACKET-2026-10-26.md (frozen Stage-1 verdict + Perplexity Search-mode survey + proposed tweaks (a)–(d)). Blind, materials only. Question: apply the tweaks and proceed to a manuscript on this literature base?

### Seat 1 — Groq API, openai/gpt-oss-20b
VERDICT (verbatim): "VERDICT: ADJUST. FATAL: none. FLAGS: RooflineBench (2026) unverifiable; KernelSight-LM (2026) unverifiable; GRASPrune (ACL 2026) unverifiable. CLAIM: yes, data shows traffic model does not beat params under matched-budget, matching literature that parameter count is a proxy."

### Seat 2 — Gemini API, gemini-3.1-flash-lite
Note: first call (gemini-3.5-flash, full packet) returned empty output; seat re-run once on a compressed packet with gemini-3.1-flash-lite and answered. Not a 429/outage.
VERDICT (verbatim): "VERDICT: ADJUST. FATAL: None. FLAGS: RooflineBench 2026, KernelSight-LM 2026, GRASPrune 2026 (Unverifiable/Future-dated). CLAIM: Yes; a bounded negative result on CPU-bound weight-plus-KV traffic modeling is scientifically defensible. The manuscript requires re-scoping to account for identified traffic model failures, utilizing established benchmarks rather than the cited future-dated works."

### In-house synthesis
- Both seats ADJUST, fatal flaws none, no KILL/reject language. Both hold the bounded negative claim (a) defensible on this base and these numbers.
- REVIEWER ERROR (both seats, same root): both flagged the three 2026 papers as unverifiable; Gemini additionally called them "future-dated" and advised building on "established benchmarks rather than the cited future-dated works." That advice is NOT adopted. In-house primary-source verification (2026-10-26): RooflineBench is real — arXiv:2602.11506, submitted 12 Feb 2026; KernelSight-LM is real — arXiv:2606.28565 (Jun 2026, v2 Jul 2026), roofline kernel model with learned efficiency term as characterised; GRASPrune is real — arXiv:2604.19398 (21 Apr 2026), "Accepted to ACL 2026 Main Conference," ACL Anthology 2026.acl-long.491, and its own text confirms "we use parameter count as the budget proxy." The seats' flags reflect model knowledge limits, not fabrication. Citations may be used, with arXiv/Anthology IDs.
- ADJUDICATION: tweaks (a)–(d) CLEARED to apply. The only seat-proposed change (drop the 2026 papers) is rejected on the verification above. Gemini's "re-scoping" point is already tweak (a). Manuscript itself remains the user's frozen-report→manuscript decision; this panel clears the base, it does not start the manuscript.


### Correction — 2026-10-26, Gemini seat (late delivery)

The gemini-3.5-flash call on the FULL packet, recorded above as returning empty output, in fact completed and its output was delivered late by the runtime. It is the canonical Gemini seat (identical full materials), and it supersedes the flash-lite compressed-packet retry quoted above:

VERDICT (verbatim, Gemini API, gemini-3.5-flash, full packet): "VERDICT: PURSUE. FATAL: none. FLAGS: GRASPrune (2026), RooflineBench (2026), and KernelSight-LM (2026) are unverifiable to me. CLAIM: Yes; LOVO relative error and ΔAIC (+2.21 at cap2, -19.11 at cap4) directly support that traffic modeling fails to consistently outperform parameter count for CPU decode prediction."

Effect on synthesis: none adverse. Panel stands Groq ADJUST / Gemini PURSUE, fatal flaws none, no KILL/reject language, bounded claim defensible, tweaks (a)–(d) cleared. The earlier lite-model ADJUST was a duplicate seat created only because the first output arrived late; the in-house paper verification and adjudication above are unchanged.
