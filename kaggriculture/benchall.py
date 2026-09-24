"""Run agent vs every replayed benchmark opponent (both seats) in parallel."""
import json, sys, time, os
from multiprocessing import Pool
from bench import *

AGENT = "main.py"
def one(args):
    f, side, path = args
    from run1 import load
    small = json.load(open(f))
    m = load(path)
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(small, 1 - side)
    res, env = play(agents, small["seed"])
    return os.path.basename(f), side, res[side], res[1 - side], small["names"][1 - side], small["rewards"][side]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else AGENT
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    jobs = [(f, side, path) for f in files()[:n] for side in (0, 1)]
    t = time.time()
    with Pool(4) as p:
        out = p.map(one, jobs)
    wins = 0; ratio = []
    for name, side, me, opp, oname, orig in out:
        wins += me > opp; ratio.append(me / opp)
        print(f"{name} s{side} me={me:8.0f} opp={opp:8.0f} ({oname[:14]}) orig_seat={orig:8.0f}")
    print(f"WINS {wins}/{len(out)}  mean ratio {sum(ratio)/len(ratio):.3f}  mean me {sum(o[2] for o in out)/len(out):.0f}  mean opp {sum(o[3] for o in out)/len(out):.0f}  {time.time()-t:.0f}s")
