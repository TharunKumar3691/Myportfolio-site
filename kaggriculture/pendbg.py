import json, sys, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f)); stops = [int(x) for x in sys.argv[3].split(",")]
m = load("main.py")
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    if obs["step"] in stops:
        s = m.build_state(obs)
        c = collections.Counter()
        for pos, opt in m.MEM["plan"].items():
            t = m.tile_at(s, pos)
            st = "empty" if t is None else (t.get("kind") if isinstance(t, dict) else t)
            c[(opt, st)] += 1
        print("step", obs["step"], "h", s.hour, "money", s.money, "seeds", {k: v for k, v in s.seeds.items() if v}, "shedAnimals", {k: v for k, v in s.shed.items() if k in m.ANIMALS})
        print("   pending", dict(c)); print("   market", a["market"])
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": max(stops) + 3}, debug=True)
env.run(agents)
