import json, sys
from bench import *
from run1 import load
f = files()[int(sys.argv[1])]; side = int(sys.argv[2]); small = json.load(open(f))
m = load("main.py")
def wrapped(obs, cfg=None):
    a = m.agent(obs, cfg)
    if obs["step"] % 48 == 12:
        s = m.build_state(obs); m.forecast(s)
        free = m.free_tiles(s)
        ex = m.new_extra()
        opts = []
        for o in m.OPTIONS:
            r = m.option_eval(s, o, s.day, ex)
            if r: opts.append((o, round(r[0]), round(r[0] / r[1], 1)))
        print(f"d{s.day} money={s.money:.0f} free={len(free)} planned={len(m.MEM['plan'])} opts={opts}")
    return a
agents = [None, None]; agents[side] = wrapped; agents[1 - side] = replay(small, 1 - side)
res, env = play(agents, small["seed"]); print(res)
