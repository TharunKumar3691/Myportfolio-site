import json, sys, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); sm = json.load(open(f))
m = load("main.py")
def summarize(env, label):
    print("==", label)
    for d in range(0, 13):
        spent = collections.Counter(); earned = collections.Counter()
        for s in range(d * 24, d * 24 + 24):
            if s + 1 >= len(env.steps): break
            m0 = env.steps[s][0]["observation"]["farms"][side]["money"]; m1 = env.steps[s + 1][0]["observation"]["farms"][side]["money"]
            a = env.steps[s + 1][side]["action"] or {}
            kinds = [q[0] + ":" + str(q[1]) if len(q) > 1 else q[0] for q in (a.get("market") or []) if q]
            if m1 < m0: spent[",".join(sorted(set(k for k in kinds if not k.startswith("SELL"))))] += m0 - m1
            elif m1 > m0: earned[",".join(sorted(set(k for k in kinds if k.startswith("SELL"))))] += m1 - m0
        farm = env.steps[d * 24 + 23][0]["observation"]["farms"][side]
        c = collections.Counter()
        for row in farm["tiles"]:
            for t in row:
                if isinstance(t, dict): c[t.get("animal") or t.get("crop") or t["kind"]] += 1
        print(f"d{d} end ${farm['money']:.0f} earned={dict(earned)} spent={ {k: round(v) for k, v in spent.items()} }\n     {dict(c)}")
agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(sm, 1 - side)
_, env = play(agents, sm["seed"]); summarize(env, "MINE")
_, env2 = play([replay(sm, 0), replay(sm, 1)], sm["seed"]); summarize(env2, "TOP " + sm["names"][side])
