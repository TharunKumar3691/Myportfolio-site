# ARC Prize 2026 — ARC-AGI-3: Graph-Exploration Agent

A Kaggle submission notebook for
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).
It uses no training, runs on CPU only and needs no internet.

| Path | What it is |
|---|---|
| `notebooks/submission.ipynb` | **The notebook to submit** (generated, don't edit by hand) |
| `agent/my_agent.py` | The agent: the only source file |
| `scripts/build_notebook.py` | Puts the agent into the notebook |
| `scripts/eval_local.py` | Local evaluator on the 25 public games, using the official RHAE scoring |
| `scripts/diag.py` | Plays one game and dumps the agent's internals |
| `notebooks/kernel-metadata.json` | Kaggle kernel metadata (set your username) |

## 1. What the competition actually is

* **Interactive benchmark.** An agent plays unseen 64×64, 16-colour turn-based
  games with no instructions. The actions are `RESET`, `ACTION1-5` and `ACTION7`
  (keys, where 7 = undo) and `ACTION6` (click at x,y). Each frame gives the grid
  (sometimes several animation layers), `state` (`NOT_FINISHED` / `WIN` /
  `GAME_OVER`), `levels_completed`, `win_levels` and `available_actions`.
* **Data section.** Beyond the offline wheels and the `ARC-AGI-3-Agents`
  framework, it has no training data. The 25 public games (with human
  baselines) come from the `arc-agi` package. Scoring uses about
  **110 private games** that nobody has seen.
* **Metric, RHAE.** For each completed level the score is
  `min(1.15, (human_baseline_actions / agent_actions)²)`. Each game's score is
  the average of its level scores weighted by level index (1, 2, 3, …); levels
  not completed count as 0. The final score is the mean over games, on a
  0–100 scale. The square makes speed critical: solving a level 10× slower
  than a human earns 1% of its value.
* **How a submission runs.** You submit a notebook. On "Save & Run All" it
  only has to write a placeholder `submission.parquet`. On the competition
  rerun (`KAGGLE_IS_COMPETITION_RERUN` is set), a `gateway` sidecar serves
  the hidden games in **competition mode**: only level resets, one scorecard,
  and one `make` per game. `main.py` plays every game **in its own thread,
  all at the same time**. The gateway records the actions and writes the
  real `submission.parquet`.
* **Limits.** The run must finish in under 12 h with no internet, and you get
  5 submissions per day. To be eligible for prizes the code must be open
  source.
* **Leaderboard context** (public sources, Aug 2026). The top entries are GPU
  LLM agents (Qwen-27B-class models served with vLLM, which write code in a
  REPL) and score about 3–3.6. Well-built programmatic agents plateau at
  about 0.27 on the leaderboard. Scores from the 25 public games stop
  predicting the leaderboard once you tune for them, so only mechanisms that
  generalise help.

## 2. The agent

1. **Perception.** The agent uses the last layer of each frame. It learns
   "ticker" regions: step counters and energy bars. These show up as small,
   thin change blobs that recur in the same row or column band whatever
   action was taken, and they are masked out of the state hash. Without the
   mask, every frame looks new and the graph never repeats a state.
2. **Click candidates.** The grid is split into connected regions of one
   colour. Each region gives one click point, at the member cell closest to
   its centroid. Regions are sorted into four salience tiers: small or medium
   regions that aren't background come first, background pockets and huge
   regions last. Colours whose clicks changed the state before move up a tier.
3. **World model.** A graph where each node is the hash of the masked grid and
   each edge is a candidate action leading to the next state. The graph also
   records which actions did nothing and which caused a death.
4. **Policy.**
   * Take an untested action in the current state (best tier first).
   * Otherwise, run a breadth-first search over known edges to the nearest
     state with untested actions, and walk there.
   * If untested actions are reachable only from the level start, `RESET`.
   * Open lower-salience tiers only after the higher ones are used up
     everywhere.
   * If the whole reachable graph is used up (the level has hidden state),
     forget the graph and explore again.
5. **Across levels.** When a level is solved, the graph is rebuilt for the
   next one. What the agent learned about the game's controls carries over:
   which keys do nothing, which click colours work, and the ticker mask.
6. **Built to survive the Kaggle run** (each of these bugs would have been
   fatal there):
   * The framework keeps every frame in memory and records every action to
     disk. With 110 games and tens of thousands of actions each, that is
     hundreds of GB of RAM and tens of GB of disk. The agent keeps only a
     short frame tail and doesn't record.
   * `GameAction` members are singletons shared by every thread, so click
     coordinates are passed explicitly rather than stored on the enum.
   * Any policy exception falls back to a safe action. Repeated gateway
     errors end only that game.
   * A global 7.5 h deadline and 30k actions per game keep the run well
     inside 12 h. Per-action log lines are suppressed.

## 3. Validation (all run here)

* Local RHAE on the 25 public games: **20 levels, mean 0.498** at 3,000 actions
  per game, and **27 levels** at 15,000 actions. Fast solves: lp85 L1 in 14
  actions (baseline 17), tn36 L1 in 15 (baseline 32), r11l L1 in 23
  (baseline 22). Compute is about 3 ms per action.
* **End-to-end run of the notebook itself** (`jupyter nbconvert --execute`)
  against a local `arc_agi` gateway in **competition mode**, with `/kaggle`
  paths mirrored and the host `gateway` pointed at it. All cells ran without
  errors, all 25 games played concurrently over HTTP, and the scorecard
  closed at **0.485** after 2 minutes. The commit-mode run wrote
  `submission.parquet`.
* `pyflakes` is clean. `nbformat.validate` passes, and a check confirms the
  notebook's agent cell is byte-identical to `agent/my_agent.py`.

## 4. How to submit

1. In `notebooks/kernel-metadata.json`, replace `REPLACE_WITH_YOUR_USERNAME`
   with your Kaggle handle. Accept the competition rules on Kaggle.
2. `kaggle kernels push -p notebooks/` (or upload `submission.ipynb` in the
   Kaggle editor, attach the competition data, set Internet **off**, and use
   Accelerator **None**).
3. When the commit run is complete, open the notebook, click
   **Submit to Competition** and pick `submission.parquet`. From the CLI:
   `kaggle competitions submit arc-prize-2026-arc-agi-3 -k <user>/arc-agi-3-graph-exploration-agent -f submission.parquet -v <version> -m "graph explorer"`.

## 5. Where more score would come from

Programmatic exploration is a robust floor, but the squared efficiency term
caps it. Every approach scoring above about 1 on the leaderboard adds a local
open-weights LLM, served once with vLLM on the RTX 6000 or H100-class GPU. The
LLM reads the screen, forms goal hypotheses and writes search code. This agent
is meant to stay underneath such a model as a fallback.
