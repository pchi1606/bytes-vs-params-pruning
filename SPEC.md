# Bytes-vs-Params Pruning — Experiment Spec (v0.1, 2026-10-23)

## Claim under test
At a fixed CPU thread cap, a pruned SLM whose cuts are **allocated by
per-token byte traffic** (weights streamed during decode + KV-cache
traffic at the target context length) completes agent tasks in fewer
CPU-seconds at equal quality than a pruned SLM whose cuts are allocated
by **parameter count / standard importance pruning** — at the *same*
total parameter reduction.

## Why this might be true (and why it might be null)
- Decode is memory-bandwidth-bound: time/token ≈ bytes read / bandwidth.
  For most dense weight matrices, bytes ∝ params, so the two allocations
  mostly agree — the expected effect comes only from where they diverge:
  (1) embedding/LM-head treatment, (2) KV cache (grows with layers ×
  hidden × context; not counted in weight params), (3) layer count vs
  width trade-offs at long (agentic) context.
- Prior-art position (screened 2026-10-23): the bytes objective itself
  is textbook (I/O survey: compression techniques share the unstated
  objective "fewer bytes moved per token"). SliceGPT cuts embedding
  dimension; FLAP allocates by activation fluctuation; Minitron by
  importance scores; NIRVANA is hardware-aware via dimension alignment
  (GPU kernel multiples), not byte accounting. No controlled
  allocation-by-byte-share vs allocation-by-params comparison at a CPU
  thread cap was found. This is an incremental allocation heuristic,
  not a new paradigm — effect size is the whole game.

## Arms (base: Llama-3.2-3B, ~25% parameter reduction, Q4_K_M GGUF)
- **A — Param-allocated (control):** standard importance pruning
  (depth + width per Minitron/Sheared-LLaMA practice).
- **B — Byte-allocated:** same total param cut, allocated ∝ measured
  per-token bytes at 4k context (depth cuts weighted up for KV savings;
  embedding/head treated by actual streamed bytes).
- **C — Unpruned baseline** (quality/speed reference).
- Identical recovery: same LoRA recovery budget + data for A and B.
  No recovery-tuning advantage to either arm.

## Measurement (llama.cpp, x86 CPU — this VM for pilot; his MacBook i9 for confirmatory)
- Thread caps: 2 and 4 (primary), 6 (secondary).
- Context: 4k tokens (agentic prompts), batch 1, fixed seeds/prompts.
- Primary metric: **CPU-seconds per completed task** on a fixed task
  suite (summarise / extract / draft / tool-call formatting, ~40 tasks,
  auto-scored) at quality within tolerance of arm A.
- Secondary: decode tok/s, perplexity (WikiText-2), peak RSS.
- Host-responsiveness note: report CPU% at cap (descriptive only).

## Decision rules (pre-committed)
- **Pursue:** B beats A by ≥10% CPU-seconds/task at equal-or-better
  task score (paired, same prompts) at BOTH 2- and 4-thread caps.
- **Kill:** advantage <10%, or in only one cap, or bought with a task-
  score drop beyond tolerance (pre-set: >2 points on the suite mean).
- Null result is reportable as-is; no metric shopping after the run.

## Cost / feasibility
- Pruning + LoRA recovery: Colab T4 (free tier, verified) or VM CPU
  for depth-only fallback. Inference measurement: VM llama.cpp (present).
- Spend: $0 expected. Time: pilot ≈ 1 day of runs, checkpointed.

## Status
Spec drafted at his go-ahead (2026-10-23). Next gate: his review of
this spec → pilot. No runs started.
