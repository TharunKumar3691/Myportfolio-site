import json, collections
from multiprocessing import Pool
from bench import *
PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
def one(f):
    sm = json.load(open(f))
    res, env = play([replay(sm, 0), replay(sm, 1)], sm["seed"])
    out = []
    for p in (0, 1):
        tot = collections.Counter(); byday = collections.defaultdict(collections.Counter)
        for s in range(719):
            o = env.steps[s][0]["observation"]; farm = o["farms"][p]
            a = env.steps[s + 1][p]["action"] or {}
            pos = [farm["farmer"]] + farm["hands"]
            acts = [a.get("farmer")] + list(a.get("hands") or [])
            seen = set()
            for i, act in enumerate(acts):
                if not act or act[0] != "HARVEST" or i >= len(pos): continue
                x, y = pos[i]
                if (x, y) in seen: continue
                seen.add((x, y))
                t = farm["tiles"][y][x]
                if isinstance(t, dict) and t.get("yield_units", 0) > 0:
                    prod = t.get("crop") or PRODUCT.get(t.get("animal"))
                    if prod:
                        tot[prod] += t["yield_units"]; byday[s // 24][prod] += t["yield_units"]
        out.append((sm["names"][p], res[p], dict(tot), {d: dict(c) for d, c in byday.items()}))
    return out
if __name__ == "__main__":
    with Pool(4) as pool:
        R = pool.map(one, files())
    agg = collections.Counter(); n = 0
    for r in R:
        for name, money, tot, byday in r:
            n += 1; agg.update(tot)
            print(f"{name[:16]:16s} {money:8.0f} {tot}")
    print("MEAN per player:", {k: round(v / n) for k, v in agg.items()})
    json.dump(R, open("/tmp/claude-0/-home-user-Myportfolio-site/fbeb2c09-8919-5f10-9b52-df19beac4385/scratchpad/prodstats.json", "w"))
