import json, collections
pool = collections.Counter(); fund = collections.Counter(); diag = collections.Counter()
cold = collections.Counter()
for line in open('results/stage1_measurements.jsonl'):
    r = json.loads(line)
    if r['kind'] != 'rep':
        continue
    if not r.get('virgin'):
        cold[(r['variant'], r['cap'])] += 1
    elif r.get('diagnostic'):
        diag[(r['variant'], r['cap'], r['ctx'])] += 1
    elif 'pair_block' in r:
        fund[(r['variant'], r['cap'], r['ctx'], r['pair_block'])] += 1
    else:
        pool[(r['variant'], r['cap'], r['ctx'])] += 1

def sess(cells):
    s = collections.Counter()
    for (v, cap, ctx), n in cells.items(): s[(v, cap)] += n
    return dict(s)

print('POOL virgin reps per session:', sess(pool))
print('pool total:', sum(pool.values()), '  pool cells complete:', sum(1 for n in pool.values() if n >= 5), '/ 84')
fc = collections.Counter()
for (v, cap, ctx, b), n in fund.items(): fc[(cap, ctx)] += n
print('FUNDING reps per (cap,ctx):', dict(fc), ' (target 10 = 5 blocks x 2 variants each)')
print('DIAGNOSTIC reps per cell:', dict(diag), ' (target 5/cell, 8 cells)')
print('pre-virgin (excluded) reps:', dict(cold))
