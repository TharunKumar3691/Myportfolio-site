import json, sys, collections
from multiprocessing import Pool
from bench import *
from run1 import load
PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
PATH = sys.argv[1] if len(sys.argv) > 1 else "main.py"
def one(args):
    f, side = args
    sm = json.load(open(f)); m = load(PATH)
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(sm, 1 - side)
    res, env = play(agents, sm["seed"])
    tot = collections.Counter(); sold = collections.Counter(); rev = collections.Counter()
    for s in range(719):
        o = env.steps[s][0]["observation"]; farm = o["farms"][side]
        a = env.steps[s + 1][side]["action"] or {}
        pos = [farm["farmer"]] + farm["hands"]; acts = [a.get("farmer")] + list(a.get("hands") or [])
        seen = set()
        for i, act in enumerate(acts):
            if not act or act[0] != "HARVEST" or i >= len(pos): continue
            x, y = pos[i]
            if (x, y) in seen: continue
            seen.add((x, y)); t = farm["tiles"][y][x]
            if isinstance(t, dict) and t.get("yield_units", 0) > 0:
                prod = t.get("crop") or PRODUCT.get(t.get("animal"))
                if prod: tot[prod] += t["yield_units"]
        # executed sales: shed decrease with money
        sh0 = env.steps[s][side]["observation"]["private"]["shed"]; sh1 = env.steps[s + 1][side]["observation"]["private"]["shed"]
        for q in a.get("market") or []:
            if q and q[0] == "SELL":
                pass
    hires = sum(len(env.steps[d*24+5][0]["observation"]["farms"][side]["hands"]) for d in range(30) if d*24+5 < len(env.steps))
    return res[side], sm["rewards"][side], dict(tot), hires
if __name__ == "__main__":
    jobs = [(f, s) for f in files()[:12] for s in (0, 1)]
    with Pool(4) as p: R = p.map(one, jobs)
    agg = collections.Counter()
    for me, orig, tot, h in R: agg.update(tot)
    n = len(R)
    print("my mean money", sum(r[0] for r in R) / n, "orig", sum(r[1] for r in R) / n, "hand-days", sum(r[3] for r in R)/n)
    print("MY MEAN units:", {k: round(v / n) for k, v in sorted(agg.items())})
    print("TOP MEAN units: {'CARROT': 169, 'EGG': 173, 'MELON': 74, 'MILK': 171, 'STRAWBERRY': 203, 'TOMATO': 126, 'WHEAT': 660, 'WOOL': 171}")
