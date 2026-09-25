import json, sys, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); sm = json.load(open(f))
m = load("main.py")
agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(sm, 1 - side)
_, env = play(agents, sm["seed"])
for d in range(1, 7):
    ops = collections.Counter(); mk = []
    for s in range(d * 24, d * 24 + 24):
        a = env.steps[s + 1][side]["action"] or {}
        for x in [a.get("farmer")] + list(a.get("hands") or []):
            if x and ("FERT" in x[0] or (len(x) > 1 and x[1] == "FERTILIZER")): ops[" ".join(map(str, x))] += 1
        for q in a.get("market") or []:
            if q and len(q) > 1 and q[1] in ("FERTILIZER", "WHEAT"): mk.append((s % 24, q[0][:4], q[1][:4], q[2]))
    pv = env.steps[d * 24 + 23][side]["observation"]["private"]
    inv = collections.Counter(); [inv.update(i) for i in pv["inventories"]]
    print(f"d{d} ops={dict(ops)} shedF={pv['shed'].get('FERTILIZER',0)} invF={inv.get('FERTILIZER',0)} price={env.steps[d*24+12][0]['observation']['market']['prices']['FERTILIZER']}\n    mk={mk}")
