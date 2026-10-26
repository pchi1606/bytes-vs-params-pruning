#!/usr/bin/env python3
"""Build the funding-attempt manifest (SPEC v0.3 Amendment Round 4
supplement, item G1, 2026-10-25).

G1 bookkeeping: every funding (pair_block) rep in
results/stage1_measurements.jsonl is identified by an immutable row
id (its line number in the append-only JSONL) plus the file md5 at
manifest time, and assigned to exactly one ATTEMPT with a status:

  valid                  complete pair, one uninterrupted adjacent
                         fresh-process attempt; retained for D1.
  invalid_restart_split  pair members straddle a VM restart
                         (D1 adjacency violation); contributes zero
                         valid pairs unless a repair attempt lands.
  incomplete_superseded  single member of an interrupted attempt,
                         superseded by a later whole-block attempt.
  incomplete_pending     single member, pair member still to come
                         in the same uninterrupted attempt.
  repaired_valid         complete whole-block repair attempt
                         (records tagged "repair": true); retained.
  lost_interrupted       attempt with NO banked reps (killed
                         mid-cell); recorded for provenance only.

A block contributes a D1 pair iff its RETAINED attempt (repair if
present, else the latest whole-block attempt, else orig) is
complete. Assignment of pre-tag records is deterministic from file
order + the coordinator's verified restart brackets (see
STAGE1-RUN-STATUS.md); records banked after the G1 amendment carry
durable "attempt" tags and are grouped by them. The builder
asserts every shape it relies on and fails loudly otherwise.
Regenerate after any phase change; analyze.py refuses to run
against a stale manifest (md5 mismatch).
"""
import hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "stage1_measurements.jsonl")
MANIFEST = os.path.join(ROOT, "results", "funding_attempts_manifest.json")
VERSION = 1

INVALID_ORIG = {(2, 8192, 1), (2, 16384, 0), (4, 16384, 0)}
# Blocks whose untagged history is [stale single member, fresh pair]:
REDO_BLOCKS = {(4, 16384, 1), (4, 16384, 2)}
# Documented exception (coordinator driver bug, 2026-10-25): blk4's
# whole-block re-run banked its two members under SPLIT attempt tags
# (f4-16384-b4-a2 on V-mixed, f4-16384-b4-a3 on V-width) because the
# attempt number was recomputed between variants. The two reps are
# one uninterrupted adjacent attempt (consecutive driver-log lines,
# fresh processes, same run); the builder merges the two singleton
# tag groups under the first tag and records the merge. The driver
# bug is fixed; no other block may use this rule.
MERGE_SPLIT_TAG = {(4, 16384, 4)}
LOST_ATTEMPTS = [
    {"cap": 2, "ctx": 8192, "pair_block": 2,
     "note": "first attempt interrupted by VM restart #2 (~07:34 UTC "
             "2026-10-25) while its first member (V-width, per the "
             "prescribed randomised order) was in flight; no rep "
             "banked. The banked pair is the subsequent attempt."},
]


def main():
    raw = open(OUT, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    rows = []
    for i, line in enumerate(raw.decode().splitlines()):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "rep" and r.get("virgin") and "pair_block" in r:
            rows.append((i, r))

    blocks = {}
    for i, r in rows:
        key = (r["cap"], r["ctx"], r["pair_block"])
        blocks.setdefault(key, []).append((i, r))

    out_blocks = []
    accounting = {"valid": 0, "invalid_restart_split": 0,
                  "incomplete_superseded": 0, "incomplete_pending": 0,
                  "repaired_valid": 0, "lost_interrupted": len(LOST_ATTEMPTS)}
    for (cap, ctx, b) in sorted(blocks):
        recs = blocks[(cap, ctx, b)]
        attempts = []
        tagged = {}
        untagged = []
        for i, r in recs:
            if r.get("attempt"):
                tagged.setdefault(r["attempt"], []).append((i, r))
            elif r.get("repair"):
                tagged.setdefault("repair-untagged", []).append((i, r))
            else:
                untagged.append((i, r))

        def mk(aid, members, status, note=None):
            att = {"attempt_id": aid,
                   "status": status,
                   "members": [{"row_id": i, "variant": r["variant"],
                                "cpu_per_token": r["cpu_per_token"]}
                               for i, r in members]}
            if note:
                att["note"] = note
            attempts.append(att)
            accounting[status] += 1

        def complete(members):
            return sorted(r["variant"] for _, r in members) == ["V-mixed", "V-width"]

        if (cap, ctx, b) in INVALID_ORIG:
            assert complete(untagged), f"{cap},{ctx},{b}: invalid orig not a full pair"
            assert len(untagged) == 2, f"{cap},{ctx},{b}: invalid orig must be one pair"
            mk(f"f{cap}-{ctx}-b{b}-orig", untagged, "invalid_restart_split")
            # tagged repair attempts (if landed) are added by the
            # generic tagged loop below -> repaired_valid.
        elif (cap, ctx, b) in REDO_BLOCKS:
            assert not tagged, f"{cap},{ctx},{b}: unexpected tagged reps"
            assert len(untagged) == 3, f"{cap},{ctx},{b}: expected 3 untagged reps"
            stale, pair = untagged[:1], untagged[1:]
            assert not complete(stale) and complete(pair), \
                f"{cap},{ctx},{b}: redo shape mismatch"
            assert stale[0][1]["variant"] == "V-mixed", \
                f"{cap},{ctx},{b}: stale member should be V-mixed (panel record)"
            mk(f"f{cap}-{ctx}-b{b}-orig", stale, "incomplete_superseded")
            mk(f"f{cap}-{ctx}-b{b}-redo1", pair, "valid")
        else:
            if untagged:
                if complete(untagged):
                    assert len(untagged) == 2, f"{cap},{ctx},{b}: >2 untagged reps"
                    mk(f"f{cap}-{ctx}-b{b}-orig", untagged, "valid")
                else:
                    assert len(untagged) == 1, f"{cap},{ctx},{b}: bad untagged shape"
                    mk(f"f{cap}-{ctx}-b{b}-orig", untagged,
                       "incomplete_superseded" if tagged else "incomplete_pending")
        merge_note = None
        if (cap, ctx, b) in MERGE_SPLIT_TAG and len(tagged) == 2:
            (t1, m1), (t2, m2) = sorted(tagged.items())
            assert len(m1) == 1 and len(m2) == 1 and \
                m1[0][1]["variant"] != m2[0][1]["variant"], \
                f"{cap},{ctx},{b}: split-tag merge shape mismatch"
            tagged = {t1: sorted(m1 + m2, key=lambda x: x[0])}
            merge_note = (f"MERGED split tags {t1} + {t2}: one uninterrupted "
                          "adjacent attempt whose records carry two attempt "
                          "tags due to a coordinator driver bug (attempt id "
                          "recomputed per variant; fixed 2026-10-25). "
                          "Adjacency evidence (H2): JSONL rows 291-294 are "
                          "prime(a2) -> rep V-mixed 0.5763 (a2) -> prime(a3) "
                          "-> rep V-width 0.6249 (a3), consecutive with no "
                          "intervening record, inside the single driver "
                          "run pid 9737 (relaunched 18:44 UTC, stopped by "
                          "the coordinator ~19:10 UTC; VM restart #9 came "
                          "later, ~19:38 UTC, under the next driver). "
                          "Corroborated by the consecutive driver-log rep "
                          "lines preserved verbatim in STAGE1-RUN-STATUS.md.")
        for aid, members in sorted(tagged.items()):
            status = ("repaired_valid" if complete(members) else "incomplete_pending") \
                if ("repair" in aid) else ("valid" if complete(members)
                                           else "incomplete_pending")
            mk(aid, members, status, note=merge_note if aid in tagged and merge_note
               and aid == sorted(tagged)[0] else None)

        # Post-pass: an incomplete attempt superseded by a later
        # complete attempt in the same block is labelled superseded,
        # not pending (e.g. repair1 interrupted, repair2 retained).
        seen_complete = False
        for a in reversed(attempts):
            if a["status"] in ("valid", "repaired_valid"):
                seen_complete = True
            elif a["status"] == "incomplete_pending" and seen_complete:
                a["status"] = "incomplete_superseded"
                accounting["incomplete_pending"] -= 1
                accounting["incomplete_superseded"] += 1
        retained = None
        for a in attempts:
            if a["status"] in ("valid", "repaired_valid"):
                retained = a["attempt_id"]  # later attempts supersede
        pair = None
        if retained:
            att = next(a for a in attempts if a["attempt_id"] == retained)
            pair = {m["variant"]: m["cpu_per_token"] for m in att["members"]}
        out_blocks.append({"cap": cap, "ctx": ctx, "pair_block": b,
                           "attempts": attempts, "retained_attempt": retained,
                           "pair": pair})

    manifest = {
        "version": VERSION,
        "scope": "all pair_block (funding + repair) reps in results/stage1_measurements.jsonl",
        "jsonl_md5_at_build": md5,
        "rule": "G1 (Round 4 supplement): a valid D1 block = exactly one V-width "
                "and one V-mixed rep from the same uninterrupted, temporally "
                "adjacent, fresh-process attempt; restart-split or incomplete "
                "attempts contribute zero valid pairs; supersession is by "
                "attempt identity in this manifest, never silent last-wins.",
        "lost_attempts": LOST_ATTEMPTS,
        "attempt_accounting": accounting,
        "blocks": out_blocks,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    n_pairs = sum(1 for bl in out_blocks if bl["pair"])
    print(f"manifest v{VERSION} written: {len(out_blocks)} blocks, "
          f"{n_pairs} retained pairs, accounting={accounting}, md5={md5}")
    for bl in out_blocks:
        print(f'  cap{bl["cap"]} ctx{bl["ctx"]} blk{bl["pair_block"]}: '
              f'retained={bl["retained_attempt"]} pair={bl["pair"]}')
    if "--assert-complete" in sys.argv:
        # H3 (Round 5): final-manifest assertions, run after the
        # repair phase and again at MEASURE DONE before analyze.py.
        per_cell = {}
        for bl in out_blocks:
            if bl["pair"]:
                per_cell[(bl["cap"], bl["ctx"])] = per_cell.get((bl["cap"], bl["ctx"]), 0) + 1
        assert n_pairs == 20, f"expected 20 retained pairs, got {n_pairs}"
        assert all(v == 5 for v in per_cell.values()) and len(per_cell) == 4, \
            f"expected 5 pairs in each of 4 funding cells, got {per_cell}"
        assert accounting["incomplete_pending"] == 0, "a funding/repair attempt is still pending"
        assert accounting["repaired_valid"] == 3, \
            f"expected 3 repaired blocks, got {accounting['repaired_valid']}"
        rows_used = [m["row_id"] for bl in out_blocks for a in bl["attempts"]
                     if a["attempt_id"] == bl["retained_attempt"] for m in a["members"]]
        assert len(rows_used) == len(set(rows_used)) == 40, "retained pairs must use 40 distinct rows"
        print("H3 completeness assertions PASSED: 20 retained pairs (5/cell), "
              "no pending attempts, 3 repaired blocks, 40 distinct rows.")


if __name__ == "__main__":
    main()
