"""Closed-loop head-to-head: agent A vs agent B on N seeds, both seat orders."""
import sys, time
from multiprocessing import Pool
from kaggle_environments import make
def one(args):
    a, b, seed, swap = args
    import importlib.util
    def ld(p):
        spec = importlib.util.spec_from_file_location("m%d" % time.time_ns(), p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    A, B = ld(a), ld(b)
    env = make("kaggriculture", configuration={"seed": 5000 + seed}, debug=True)
    env.run([B.agent, A.agent] if swap else [A.agent, B.agent])
    r = [s.reward for s in env.steps[-1]]
    return (r[1], r[0]) if swap else (r[0], r[1])
if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(4) as p:
        R = p.map(one, [(a, b, s, sw) for s in range(n) for sw in (0, 1)])
    w = sum(x > y for x, y in R); l = sum(x < y for x, y in R)
    print(f"{a} vs {b}: W{w} L{l} T{len(R)-w-l}  meanA={sum(x for x,_ in R)/len(R):.0f} meanB={sum(y for _,y in R)/len(R):.0f}")
