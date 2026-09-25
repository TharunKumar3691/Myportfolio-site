# Kaggriculture agent

Agent for the Kaggle simulation competition [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture):
a two-player, 30-day farming economy where the player with more money at the end wins.

| File | Purpose |
| --- | --- |
| `main.py` | The submission: `agent(obs, config)` (single file, no dependencies) |
| `kaggriculture-agent.ipynb` | Notebook: strategy write-up, `%%writefile main.py`, local validation, `submission.tar.gz` |
| `build_notebook.py` | Regenerates the notebook from `main.py` |
| `tools/` | Local evaluation and debugging scripts (replay benchmark, paired comparison, waste analysis) |

## How it plays

1. **Forecaster**: builds a day-by-day inventory path for every product from remaining town demand
   (known shops plus expected future shops) and the projected output of both farms. A unit sold on
   day *d* is valued at the price on that path.
2. **Planner**: greedily assigns free tiles to the crop or animal with the best discounted net value
   per unit of cash plus tile-days, with diminishing returns as commitments are added. It uses the
   consensus opening (2 cows, 3 sheep, melons, wheat) and plants strawberries from day 2 to day 5.
3. **Scheduler**: turns tiles into valued tasks (water, feed, care, harvest, fertilize, collect,
   plant, build, place, dig) with same-day urgency. It matches the farmer and hands to tasks each
   turn, including wheat, fertilizer and animal pickups, and shed deliveries (selling cash early,
   staying under the 100-item shed cap).
4. **Market**: sells produce promptly, keeps feed and fertilizer reserves, buys feed wheat when the
   shed runs low, and hires at most 12 hands a day on the Fibonacci cost curve.

## Local results

Benchmark: 63 recent episodes of the top-20 leaderboard teams. Their recorded actions are replayed
with the episode seed, and our agent takes one seat (both seats tested, 126 games).

* Mean final money: **$103.5k**. The top agents' own result from the same seats averages ~$102.8k
  (ratio 1.007).
* Head-to-head against earlier versions of this agent: 20–0.
* Runtime: about 2–15 ms per turn (limit is 1 s).

Caveat: replayed opponents are open-loop (they cannot react to us), so these are estimates of
strength, not leaderboard guarantees.

## Reproduce

```bash
pip install kaggle-environments
python tools/crashtest.py 12              # full games vs self/starter/random on fresh seeds
BENCH=/path/to/bench python tools/compare.py main.py   # replay benchmark (needs downloaded episodes)
```

## Submit

```bash
kaggle competitions submit kaggriculture -f main.py -m "value-driven agent"
```
