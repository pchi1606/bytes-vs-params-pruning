#!/usr/bin/env python3
"""Deterministic scorer for the 40-task suite.
Usage: score.py tasks.json outputs_dir  -> prints per-task 0/1 and mean score (points = mean*100).
outputs_dir contains one file per task id: <id>.txt with the raw model output."""
import json, re, sys, os

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower().strip(" .\"'`"))

def check(chk, out):
    kind = chk["kind"]
    if kind == "exact":
        return 1.0 if norm(out) == norm(str(chk["answer"])) else 0.0
    if kind == "contains":
        return 1.0 if norm(chk["answer"]) in norm(out) else 0.0
    if kind == "number":
        nums = re.findall(r"-?\d+", out)
        return 1.0 if nums and int(nums[-1]) == chk["answer"] else 0.0
    if kind == "bullets":
        lines = [l for l in out.strip().splitlines() if l.strip().startswith("- ")]
        return 1.0 if len(lines) == chk["n"] else 0.0
    if kind == "json_keys":
        m = re.search(r"\{.*\}", out, re.S)
        if not m: return 0.0
        try: obj = json.loads(m.group(0))
        except Exception: return 0.0
        if sorted(obj.keys()) != sorted(chk["keys"]): return 0.0
        for k, v in chk.get("values", {}).items():
            if str(obj.get(k)).lower() != str(v).lower(): return 0.0
        return 1.0
    if kind == "json_array":
        m = re.search(r"\[.*\]", out, re.S)
        if not m: return 0.0
        try: arr = json.loads(m.group(0))
        except Exception: return 0.0
        return 1.0 if arr == chk["answer"] else 0.0
    if kind == "word_count":
        return 1.0 if len(out.strip().split()) == chk["n"] else 0.0
    if kind == "word_count_max":
        return 1.0 if 0 < len(out.strip().split()) <= chk["n"] else 0.0
    if kind == "keywords":
        o = norm(out)
        if len(out.strip().split()) > chk["max_words"]: return 0.0
        return 1.0 if all(k.lower() in o for k in chk["keywords"]) else 0.0
    raise ValueError(kind)

def main():
    tasks = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    scores = {}
    for t in tasks:
        p = os.path.join(outdir, t["id"] + ".txt")
        out = open(p).read() if os.path.exists(p) else ""
        scores[t["id"]] = check(t["check"], out)
    mean = sum(scores.values()) / len(scores)
    by_type = {}
    for t in tasks:
        by_type.setdefault(t["type"], []).append(scores[t["id"]])
    print(json.dumps({"mean_points": round(mean * 100, 2),
                      "by_type": {k: round(sum(v) / len(v) * 100, 1) for k, v in by_type.items()},
                      "per_task": scores}, indent=1))

if __name__ == "__main__":
    main()
