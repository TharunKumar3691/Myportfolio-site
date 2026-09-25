import json, sys, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
m = load("main.py")
agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
res, env = play(agents, small["seed"])
for d in range(30):
    s = d * 24 + 23
    if s >= len(env.steps): break
    farm = env.steps[s][0]["observation"]["farms"][side]
    un = []
    for y in range(10):
        for x in range(10):
            t = farm["tiles"][y][x]
            if isinstance(t, dict) and t.get("animal") and not t["fed_today"]:
                un.append(((x, y), t["animal"][0], t["placed_day"], t["consecutive_unfed"], t["cared_today"]))
    if un: print(d, un)
