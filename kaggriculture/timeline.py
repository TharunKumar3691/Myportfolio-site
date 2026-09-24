import json, sys
from multiprocessing import Pool
from bench import *
from run1 import load
PATH = sys.argv[1]
DAYS = [5, 8, 10, 12, 15, 18, 21, 24, 27, 29]
def money_days(env, side):
    return [env.steps[min(d * 24 + 23, len(env.steps) - 1)][0]["observation"]["farms"][side]["money"] for d in DAYS]
def one(args):
    f, side = args
    sm = json.load(open(f)); m = load(PATH)
    agents = [None, None]; agents[side] = m.agent; agents[1 - side] = replay(sm, 1 - side)
    _, env = play(agents, sm["seed"])
    _, env2 = play([replay(sm, 0), replay(sm, 1)], sm["seed"])
    return money_days(env, side), money_days(env2, side)
if __name__ == "__main__":
    jobs = [(f, s) for f in files()[:15] for s in (0, 1)]
    with Pool(4) as p: R = p.map(one, jobs)
    n = len(R)
    print("day   ", DAYS)
    print("mine  ", [round(sum(r[0][i] for r in R) / n) for i in range(len(DAYS))])
    print("top   ", [round(sum(r[1][i] for r in R) / n) for i in range(len(DAYS))])
