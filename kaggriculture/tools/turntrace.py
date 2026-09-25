import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
d0, d1 = int(sys.argv[3]), int(sys.argv[4])
m = load("main.py")
log = []
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    st = obs["step"]
    if d0 * 24 <= st < d1 * 24:
        s = m.build_state(obs); m.forecast(s)
        tasks = m.gen_tasks(s)
        tl = {p: [(t.op, round(t.val)) for t in l] for p, l in tasks.items()}
        units = [(u["pos"], {k: v for k, v in u["inv"].items()}) for u in s.units]
        print(f"step {st} d{s.day} h{s.hour} money {s.money:.0f}")
        print("   units", [(u, a2) for u, a2 in zip(units, [a["farmer"]] + a["hands"])])
        print("   tasks", tl)
        print("   market", a["market"])
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": d1 * 24 + 2}, debug=True)
env.run(agents)
