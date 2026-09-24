import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f)); a0, a1 = int(sys.argv[3]), int(sys.argv[4])
m = load("main.py")
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    if a0 <= obs["step"] < a1:
        s = m.build_state(obs)
        acts = [a["farmer"]] + a["hands"]
        rel = [(u["idx"], u["pos"], {k: v for k, v in u["inv"].items() if k in m.ANIMALS}, acts[u["idx"]], m.MEM["targets"].get(u["idx"])) for u in s.units if any(k in m.ANIMALS for k in u["inv"]) or acts[u["idx"]][0] in ("PICKUP", "PLACE")]
        print(obs["step"], "h", s.hour, "shed animals", {k: v for k, v in s.shed.items() if k in m.ANIMALS}, rel)
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
from kaggle_environments import make
env = make("kaggriculture", configuration={"seed": small["seed"], "episodeSteps": a1 + 2}, debug=True)
env.run(agents)
