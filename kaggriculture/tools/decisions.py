import json, sys, collections
from multiprocessing import Pool
from bench import *
from run1 import load
PATH = sys.argv[1]
def feats(env, side):
    out = {}
    land = {}
    for s in range(0, len(env.steps), 6):
        q = env.steps[s][0]["observation"]["farms"][side]["unlocked_quadrants"]
        for i in range(2, 5):
            if len(q) >= i and i not in land: land[i] = s // 24
    for i in range(2, 5): out[f"land{i}"] = land.get(i, 30)
    for d in (4, 8, 12, 16):
        c = collections.Counter()
        for row in env.steps[d * 24 + 12][0]["observation"]["farms"][side]["tiles"]:
            for t in row:
                if isinstance(t, dict): c[t.get("animal") or t.get("crop") or t["kind"]] += 1
        for k in ("COW", "SHEEP", "GOOSE", "STRAWBERRY", "MELON", "TOMATO", "WHEAT"):
            out[f"{k[:5]}@{d}"] = c.get(k, 0)
    return out
def one(args):
    f, side = args
    sm = json.load(open(f)); m = load(PATH)
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(sm, 1 - side)
    _, env = play(agents, sm["seed"]); _, env2 = play([replay(sm, 0), replay(sm, 1)], sm["seed"])
    return feats(env, side), feats(env2, side)
if __name__ == "__main__":
    jobs = [(f, s) for f in files()[:10] for s in (0, 1)]
    with Pool(4) as p: R = p.map(one, jobs)
    keys = list(R[0][0].keys())
    for k in keys:
        print(f"{k:14s} mine={sum(r[0][k] for r in R)/len(R):6.1f}  top={sum(r[1][k] for r in R)/len(R):6.1f}")
