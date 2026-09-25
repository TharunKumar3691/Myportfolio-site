import json, sys, collections
from bench import *
from run1 import load
def analyze(env, side):
    W = collections.Counter()
    steps = env.steps
    for s in range(1, len(steps)):
        prev = steps[s-1][0]["observation"]["farms"][side]["tiles"]
        cur = steps[s][0]["observation"]["farms"][side]["tiles"]
        eod = (s % 24 == 0)
        for y in range(10):
            for x in range(10):
                a, b = prev[y][x], cur[y][x]
                if isinstance(a, dict) and a.get("kind") == "PLANT" and isinstance(b, dict) and b.get("kind") == "WEED":
                    W["plant->weed(" + a["crop"] + ",y=%d)" % a.get("yield_units", 0)] += 1
                if isinstance(a, dict) and a.get("animal") and isinstance(b, dict) and not b.get("animal"):
                    W["escaped " + a["animal"]] += 1
                if eod and isinstance(a, dict) and a.get("animal"):
                    if not a.get("fed_today"): W["unfed-day " + a["animal"]] += 1
                    if not a.get("cared_today"): W["uncared-day " + a["animal"]] += 1
                if isinstance(a, dict) and a.get("kind") == "PLANT" and isinstance(b, dict) and b.get("kind") == "PLANT" and b.get("yield_units", 0) < a.get("yield_units", 0) and not b.get("watered_today", False) == False:
                    pass
        # decay losses
        for y in range(10):
            for x in range(10):
                a, b = prev[y][x], cur[y][x]
                if isinstance(a, dict) and isinstance(b, dict) and a.get("kind") == "PLANT" and b.get("kind") in ("PLANT", "WEED"):
                    if b.get("kind") == "PLANT" and b.get("yield_units", 0) < a.get("yield_units", 0) and (a.get("yield_units",0) - b.get("yield_units",0)) == 1 and b.get("crop") == a.get("crop"):
                        # harvest resets to 0; decay subtracts exactly 1
                        W["decay " + a["crop"]] += 1
        if eod:
            sh = steps[s-1][side]["observation"]["private"]["shed"]
            inv = steps[s-1][side]["observation"]["private"]["inventories"]
            tot = sum(sh.values()) + sum(sum(i.values()) for i in inv)
            if tot > 100: W["overflow_units"] += tot - 100
    return W
if __name__ == "__main__":
    f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
    m = load(sys.argv[3] if len(sys.argv) > 3 else "main.py")
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
    res, env = play(agents, small["seed"])
    print(res)
    print("ME ", dict(analyze(env, side)))
    print("OPP", dict(analyze(env, 1 - side)))
