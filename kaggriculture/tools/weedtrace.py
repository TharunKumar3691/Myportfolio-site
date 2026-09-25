import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
m = load("main.py")
agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
res, env = play(agents, small["seed"])
n = 0
for s in range(1, len(env.steps)):
    prev = env.steps[s-1][0]["observation"]["farms"][side]["tiles"]; cur = env.steps[s][0]["observation"]["farms"][side]["tiles"]
    for y in range(10):
        for x in range(10):
            a, b = prev[y][x], cur[y][x]
            if isinstance(a, dict) and a.get("kind") == "PLANT" and isinstance(b, dict) and b.get("kind") == "WEED" and a.get("yield_units") == 1:
                n += 1
                if n > 4: continue
                print("weed at step", s, (x, y), a)
                # actions on that tile during planted day
                pd = a["planted_day"]
                for t in range(pd * 24, s):
                    o = env.steps[t][0]["observation"]["farms"][side]
                    act = env.steps[t + 1][side]["action"] or {}
                    pos = [tuple(o["farmer"])] + [tuple(h) for h in o["hands"]]
                    acts = [act.get("farmer")] + list(act.get("hands") or [])
                    for i, p in enumerate(pos):
                        if p == (x, y) and i < len(acts) and acts[i] and acts[i][0] not in ("NORTH","SOUTH","EAST","WEST"):
                            print("   t", t, "d", t // 24, "h", t % 24, "unit", i, acts[i])
print("total", n)
