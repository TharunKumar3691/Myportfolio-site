"""Local evaluation harness for Kaggriculture agents."""
import json, time, sys, importlib.util
from kaggle_environments import make

def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_mod_%d" % abs(hash(path + str(time.time()))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.agent

def replay_agent(replay_path, player):
    r = json.load(open(replay_path))
    steps = r["steps"]
    def agent(obs, cfg=None):
        t = obs["step"] if isinstance(obs, dict) else obs.step
        if t + 1 < len(steps):
            return steps[t + 1][player]["action"] or {}
        return {}
    return agent, r["info"]["seed"], r["rewards"]

def run(agents, seed=None, steps=720):
    cfg = {"episodeSteps": steps}
    if seed is not None: cfg["seed"] = seed
    env = make("kaggriculture", configuration=cfg, debug=True)
    t = time.time()
    env.run(agents)
    final = env.steps[-1]
    return [s.reward for s in final], time.time() - t, env

if __name__ == "__main__":
    rp = sys.argv[1]
    a0, seed, rw = replay_agent(rp, 0); a1, _, _ = replay_agent(rp, 1)
    res, dt, _ = run([a0, a1], seed)
    print("reproduced", res, "recorded", rw, "time %.1fs" % dt)
