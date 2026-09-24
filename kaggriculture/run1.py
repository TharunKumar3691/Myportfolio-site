import json, sys, time
from bench import *
import importlib.util
def load(path):
    spec = importlib.util.spec_from_file_location("m%d" % time.time_ns(), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
if __name__ == "__main__":
    f = files()[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    side = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    small = json.load(open(f))
    m = load(sys.argv[3] if len(sys.argv) > 3 else "main.py")
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
    t = time.time(); res, env = play(agents, small["seed"])
    print(f, "me(side %d)=%.0f opp(%s)=%.0f  [orig: %s]  %.1fs" % (side, res[side], small["names"][1-side], res[1-side], small["rewards"], time.time()-t))
