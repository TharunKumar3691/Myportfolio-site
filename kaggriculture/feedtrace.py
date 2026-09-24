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
    o = env.steps[s][0]["observation"]; farm = o["farms"][side]
    pv = env.steps[s][side]["observation"]["private"]
    an = [t for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal")]
    unfed = sum(1 for t in an if not t["fed_today"])
    carried = sum(i.get("WHEAT", 0) for i in pv["inventories"])
    print(f"d{d} h23 money={farm['money']:.0f} animals={len(an)} unfed={unfed} shedWheat={pv['shed'].get('WHEAT',0)} carriedWheat={carried} shedTotal={sum(pv['shed'].values())} wheatPrice={o['market']['prices']['WHEAT']} units={1+len(farm['hands'])}")
