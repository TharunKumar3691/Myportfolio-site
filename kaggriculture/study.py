import json, sys, collections
from bench import *
small = json.load(open(BENCH + "/%s.json" % sys.argv[1])); P = int(sys.argv[2])
res, env = play([replay(small, 0), replay(small, 1)], small["seed"])
print(res, small["names"])
for s in range(0, 720, 48):
    o = env.steps[s][0]["observation"]; f = o["farms"][P]
    c = collections.Counter()
    for row in f["tiles"]:
        for t in row:
            if isinstance(t, dict): c[t.get("animal") or t.get("crop") or t["kind"]] += 1
    print(f"d{o['day']} ${f['money']:.0f} q={len(f['unlocked_quadrants'])} {dict(c)}")
