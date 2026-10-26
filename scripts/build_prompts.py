#!/usr/bin/env python3
"""Fixed nested filler prompts: one token stream, prefixes at each
occupied-context length (amendment A4: fixed token IDs)."""
import json, os
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = open(os.path.join(ROOT, "base_path.txt")).read().strip()
tok = AutoTokenizer.from_pretrained(BASE)
para = ("The river valley stretches from the northern hills to the southern plain, "
        "crossed by roads that connect small towns and farming villages. In spring the "
        "fields are planted with wheat and barley, and the markets in the county town "
        "fill with traders from the surrounding districts. The old railway line follows "
        "the river for many miles before turning east toward the coast, passing through "
        "orchards, meadows, and stands of oak and pine. Local records describe floods, "
        "harvests, and the building of bridges, mills, and schools over two centuries. ")
ids = []
while len(ids) < 17000:
    ids += tok(para, add_special_tokens=False)["input_ids"]
out = {str(n): ids[:n] for n in (512, 2048, 4096, 6144, 8192, 12288, 16384)}
json.dump(out, open(os.path.join(ROOT, "results", "prompts.json"), "w"))
print({k: len(v) for k, v in out.items()})
