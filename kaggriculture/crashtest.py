import sys, io, contextlib, time
from multiprocessing import Pool
from kaggle_environments import make
def one(seed):
    import importlib.util
    def ld():
        spec = importlib.util.spec_from_file_location("m%d" % time.time_ns(), "main.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    a, b = ld(), ld()
    opp = [b.agent, "starter", "random", "pass"][seed % 4]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        env = make("kaggriculture", configuration={"seed": 1000 + seed}, debug=True)
        env.run([a.agent, opp])
    out = buf.getvalue()
    return seed, [s.reward for s in env.steps[-1]], [s.status for s in env.steps[-1]], out.count("Traceback"), out[:600] if "Traceback" in out else ""
if __name__ == "__main__":
    with Pool(4) as p:
        for r in p.map(one, range(int(sys.argv[1]) if len(sys.argv) > 1 else 12)):
            print(r[0], r[1], r[2], "tracebacks:", r[3]); 
            if r[4]: print(r[4])
