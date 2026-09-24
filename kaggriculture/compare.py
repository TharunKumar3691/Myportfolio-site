"""Paired comparison of agent versions on the replay benchmark.
usage: python compare.py a.py [b.py ...]   (results cached in results/<name>.json)"""
import json, sys, os, time, math, hashlib
from multiprocessing import Pool
from bench import *
from benchall import one

def run_agent(path, jobs_files):
    key = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
    os.makedirs("results", exist_ok=True)
    cache = f"results/{os.path.basename(path)}.{key}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    jobs = [(f, side, path) for f in jobs_files for side in (0, 1)]
    with Pool(4) as p:
        out = p.map(one, jobs)
    res = {f"{o[0]}:{o[1]}": [o[2], o[3], o[5]] for o in out}
    json.dump(res, open(cache, "w"))
    return res

if __name__ == "__main__":
    fl = files()
    allr = {}
    for path in sys.argv[1:]:
        t = time.time()
        r = run_agent(path, fl)
        allr[path] = r
        me = [v[0] for v in r.values()]; wins = sum(v[0] > v[1] for v in r.values()); orig = [v[2] for v in r.values()]
        print(f"{path:28s} n={len(me)} mean={sum(me)/len(me):8.0f}  wins={wins}/{len(me)}  vs-orig-seat={sum(a/b for a,b in zip(me,orig))/len(me):.3f}  ({time.time()-t:.0f}s)")
    base = sys.argv[1]
    for path in sys.argv[2:]:
        d = [allr[path][k][0] - allr[base][k][0] for k in allr[base] if k in allr[path]]
        m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        print(f"  {path} - {base}: {m:+.0f} ± {sd/math.sqrt(len(d)):.0f}")
