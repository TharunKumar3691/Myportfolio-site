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
    pv = env.steps[s][side]["observation"]["private"]
    inv = collections.Counter()
    for i in pv["inventories"]: inv.update(i)
    a = env.steps[s + 1][side]["action"] if s + 1 < len(env.steps) else {}
    print(f"d{d} shed={ {k:v for k,v in pv['shed'].items() if v} } inv={dict(inv)} mk={a.get('market')}")
