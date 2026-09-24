import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
d0 = int(sys.argv[3])
m = load("main.py")
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    st = obs["step"]
    if d0 * 24 <= st < (d0 + 1) * 24:
        s = m.build_state(obs); m.forecast(s); tasks = m.gen_tasks(s)
        feeds = [(p, round(t.val)) for p, l in tasks.items() for t in l if t.op == "FEED"]
        wu = [(u["idx"], u["pos"], u["inv"].get("WHEAT", 0)) for u in s.units if u["inv"].get("WHEAT", 0)]
        acts = [a["farmer"]] + a["hands"]
        feeders = [(i, x) for i, x in enumerate(acts) if x[0] in ("FEED", "PICKUP")]
        print(f"h{s.hour} shedW={s.shed.get('WHEAT',0)} units={len(s.units)} wheatUnits={wu} FEEDtasks={feeds} feed/pick={feeders} mk={[o for o in a['market'] if 'WHEAT' in o]}")
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": (d0 + 1) * 24 + 2}, debug=True)
env.run(agents)
st = env.steps[d0 * 24][0]["observation"]["farms"][side]["tiles"]
for (x, y) in [(4, 0), (3, 3), (3, 4)]:
    print((x, y), st[y][x])
