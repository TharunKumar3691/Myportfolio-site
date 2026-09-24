import json, sys, time, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
m = load(sys.argv[3] if len(sys.argv) > 3 else "main.py")
agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
res, env = play(agents, small["seed"])
print("result", res, "orig", small["rewards"])
def comp(tiles):
    c = collections.Counter()
    for row in tiles:
        for t in row:
            if isinstance(t, dict): c[t.get("animal") or t.get("crop") or t["kind"]] += 1
            elif t is None: c["empty"] += 1
    return dict(c)
for d in range(30):
    s0 = d * 24
    o = env.steps[s0][0]["observation"]; f0 = o["farms"][side]; fo = o["farms"][1-side]
    ops = collections.Counter(); mk = collections.Counter(); maxh = 0
    for s in range(s0 + 1, min(s0 + 25, len(env.steps))):
        a = env.steps[s][side]["action"] or {}
        for x in [a.get("farmer")] + list(a.get("hands", []) or []):
            if x: ops[x[0]] += 1
        for q in a.get("market", []) or []:
            mk[q[0] + ("" if len(q) < 2 else ":" + str(q[1]))] += (q[2] if len(q) > 2 else 1)
        maxh = max(maxh, len(env.steps[s][0]["observation"]["farms"][side]["hands"]))
    priv = env.steps[s0][side]["observation"]["private"]
    print(f"d{d} me ${f0['money']:.0f} opp ${fo['money']:.0f} q={len(f0['unlocked_quadrants'])} hands={maxh} {comp(f0['tiles'])}")
    if len(sys.argv) > 4:
        print("   shed", {k: v for k, v in priv["shed"].items() if v}, "seeds", {k: v for k, v in priv["seeds"].items() if v})
        print("   ops", dict(ops.most_common()))
        print("   mk", dict(mk))
