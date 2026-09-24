import json, glob, os, sys, time
from kaggle_environments import make
BENCH = os.environ.get("BENCH", "/tmp/claude-0/-home-user-Myportfolio-site/fbeb2c09-8919-5f10-9b52-df19beac4385/scratchpad/bench")

def replay(small, p):
    acts = small["actions"]
    def agent(obs, cfg=None):
        t = obs["step"]
        return (acts[t + 1][p] if t + 1 < len(acts) else None) or {}
    return agent

def play(agents, seed):
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.run(agents)
    return [s.reward for s in env.steps[-1]], env

def files():
    return sorted(glob.glob(BENCH + "/*.json"))
