import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f)); stop = int(sys.argv[3])
m = load("main.py")
snap = {}
def wrapped(obs, cfg=None):
    if obs["step"] == stop:
        s = m.build_state(obs); m.forecast(s)
        snap["s"] = s
        print("day", s.day, "hour", s.hour, "money", s.money, "shops", s.shops)
        print("prices", s.prices)
        print("plan", sorted(m.MEM["plan"].items()))
        print("shed", s.shed, "seeds", s.seeds)
        for p in m.PRODUCTS:
            print("  path", p, [m.price(p, int(s.path[p][d])) for d in range(s.day, 30, 2)])
        ex = m.new_extra()
        for o in m.OPTIONS:
            r = m.option_eval(s, o, s.day, ex)
            if r: print("  opt", o, "value", round(r[0]), "occ", r[1], "cost", r[3], "score", round(r[0] / (r[3] + r[1] * m.P["tile_shadow"]), 2), "rate/day", round(r[0] / r[1], 1))
    return m.agent(obs, cfg)
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": stop + 3}, debug=True)
env.run(agents)
