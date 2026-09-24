"""Build kaggriculture-agent.ipynb from main.py (single source of truth)."""
import json

agent_src = open("main.py").read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# Kaggriculture — value-driven farming agent

Two-player, 30-day (720-turn) farming economy. Score = bank balance at the end; the
leaderboard rating comes from head-to-head wins, so the goal is simply **more money than the
opponent** in every episode.

## What the game rewards (from reading the engine + top-agent replays)

* **Shared, thin premium markets.** Each product's price is a function of *market inventory*
  relative to 10,000. Town shops drain inventory (price up), both players' sales add to it
  (price down). Melon / milk / wool / strawberry prices collapse to $1 on a glut, so value comes
  from selling into *town demand*, which differs per episode (random shop draws).
* **Animal care bonus.** Feeding **and** caring every day banks +1 unit for the next production.
  A sheep's first yield is 6 wool (~$1,300), then 4 wool every 3 days; cows give 6 then 3 milk
  every 2 days. Missing feed or care loses that bonus.
* **Fertilizer** (1/animal/day) sells for ~$100 early and doubles strawberry/tomato output;
  its price only goes down over the game as both players sell it.
* **Cash compounds early.** $3,000 start; the first ~10 days are cash-constrained.
* **The final turn is step 718 (day 29, hour 22)**: the day-29 end-of-day never happens, so
  goods must be dropped at the shed and sold before then.

## Agent architecture (`main.py`)

1. **Forecaster** – builds, for every product, the day-by-day market inventory path from
   remaining town demand (known shops + expected future shops) and projected output of *both*
   farms (the opponent's farm is public). A unit sold on day *d* is valued at the price on that
   path, so the agent sees gluts coming.
2. **Planner** – greedily assigns free tiles to the crop/animal with the best discounted net
   value per (cash + tile-day), with diminishing returns as commitments are added (and an
   assumption that the opponent mirrors part of our expansion). Buys seeds, animals and land.
3. **Scheduler** – converts every tile into valued tasks (water, feed, care, harvest,
   fertilize, collect fertilizer, plant, build, place, dig) with same-day urgency, and matches
   the farmer + hired hands to tasks each turn, including supply trips (wheat, fertilizer,
   animals) and shed deliveries (sell cash early, avoid the 100-item shed overflow).
4. **Market** – sells produce immediately, keeps feed/fertilizer reserves, buys feed wheat when
   the shed runs low, hires hands for the day's workload on the Fibonacci cost curve.

Evaluation was done locally against replays of the current top-20 leaderboard agents
(same episode seed, their recorded actions replayed), in both seats.
"""),
    code("""
%%writefile main.py
""" + agent_src),
    md("## Local validation\nRuns full 720-turn games against the built-in agents and against itself, checks for errors and per-turn time."),
    code("""
import time, importlib, sys
from kaggle_environments import make

sys.path.insert(0, ".")
import main as agent_module
importlib.reload(agent_module)

for opp in ["starter", "random", agent_module.agent]:
    env = make("kaggriculture", debug=True)
    t = time.time()
    env.run([agent_module.agent, opp])
    final = env.steps[-1]
    name = opp if isinstance(opp, str) else "self"
    print(f"vs {name:8s} rewards={[s.reward for s in final]} statuses={[s.status for s in final]} "
          f"({(time.time() - t) / 720 * 1000:.1f} ms/turn)")
"""),
    md("## Submission file\nThe competition expects `main.py` with an `agent` function at the archive root."),
    code("""
!tar -czf submission.tar.gz main.py
!ls -la main.py submission.tar.gz
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open("kaggriculture-agent.ipynb", "w"), indent=1)
print("wrote kaggriculture-agent.ipynb")
