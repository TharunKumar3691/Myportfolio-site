import json, glob, os
from multiprocessing import Pool
from bench import replay, play
from run1 import load
T = "/tmp/claude-0/-home-user-Myportfolio-site/fbeb2c09-8919-5f10-9b52-df19beac4385/scratchpad/tiers"
def one(f):
    sm = json.load(open(f)); rs = sm["side"]; me = 1 - rs
    m = load("../main.py")
    agents = [None, None]; agents[me] = m.agent; agents[rs] = replay(sm, rs)
    res, _ = play(agents, sm["seed"])
    return os.path.basename(os.path.dirname(f)), res[me], res[rs], sm["rewards"][me]
if __name__ == "__main__":
    fs = sorted(glob.glob(T + "/*/*.json"))
    with Pool(4) as p: R = p.map(one, fs)
    from collections import defaultdict
    g = defaultdict(list)
    for t, a, b, o in R: g[t].append((a, b, o))
    for t in sorted(g, key=lambda x: int(x[1:])):
        v = g[t]
        print(f"rank {t[1:]:>4}: wins {sum(a > b for a, b, _ in v)}/{len(v)}  my mean ${sum(a for a,_,_ in v)/len(v):,.0f}  their mean ${sum(b for _,b,_ in v)/len(v):,.0f}  (their original opponent earned ${sum(o for *_, o in v)/len(v):,.0f})")
