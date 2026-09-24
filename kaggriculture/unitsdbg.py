import json, sys, collections
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f)); stops = [int(x) for x in sys.argv[3].split(",")]
m = load("main.py")
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    if obs["step"] in stops:
        s = m.build_state(obs); m.forecast(s); tasks = m.gen_tasks(s)
        acts = [a["farmer"]] + a["hands"]
        print("step", obs["step"], "h", s.hour)
        for u in s.units:
            print("   u", u["idx"], u["pos"], {k: v for k, v in u["inv"].items()}, acts[u["idx"]], "target", m.MEM["targets"].get(u["idx"]))
        top = sorted(((round(t.val), t.op, p) for p, l in tasks.items() for t in l), reverse=True)[:40]
        print("   top tasks", top)
        c = collections.Counter(t.op for l in tasks.values() for t in l); print("   task counts", dict(c))
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": max(stops) + 3}, debug=True)
env.run(agents)
