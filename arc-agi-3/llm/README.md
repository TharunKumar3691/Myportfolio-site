# Hybrid LLM notebook (GPU): Duck harness + Qwen3.8 + graph explorer

`notebook/arc3-hybrid-llm.ipynb` is the GPU submission for ARC Prize 2026 — ARC-AGI-3.
`notebook/kernel-metadata.json` pushes it as `tharunkumar369/arc-agi-3-hybrid-llm-agent`
on an **RTX Pro 6000** with internet off.

## What runs

| Part | Source | Why |
|---|---|---|
| Solver loop | Tufa Labs **Duck** harness ([github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness), MIT, commit `7652836`) | Milestone #1 winner. The LLM writes Python in a sandbox that sees the board (image + segmentation) and calls `action(...)`. |
| Model | `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` (Qwen3.8-27B-FP8) | Moving from Qwen3.6 to Qwen3.8 gave +0.63 on the leaderboard in a public experiment log |
| Board images | `MULTIMODAL_UPSCALE=8` (512×512) | +1.03 on the leaderboard in the same log (1.53 → 2.56) |
| KV cache | `--kv-cache-dtype fp8` | Fits the larger image context in memory |
| vLLM | `driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM 0.19, torch 2.10) | The wheelhouse the Duck notebooks use |
| **Hybrid opening** (ours) | `inference/agent/graph_opening.py` | On **click-only** games a graph explorer gets ≤24 clicks on level 1 before the LLM takes over. It demotes HUD/timer strips and uses BFS over the observed state graph. `HYBRID_GRAPH_OPENING=0` turns it off. |
| **Fallback** (ours) | `../agent/my_agent.py` | If vLLM cannot start during the scored rerun, the CPU graph agent plays every game instead of the run scoring 0 |

The harness source and pickled solver/benchmark ("TAAF source bundle") are
embedded in the notebook as base64 (`taaf_bundle.tar.gz`, 420 KB). No private
dataset upload is needed. Our changes to the upstream harness are in
`duck-upgrades.patch`.

**Save & Run All** (the commit run) does only a smoke run: it starts vLLM,
plays 2 public games for at most 25 minutes, and writes a placeholder
`submission.parquet`. The full 110-game run happens in the competition rerun
after **Submit to Competition**. That run uses the harness budgets: 132 min
per game, 28 games at a time, and 9 h overall.

## How the hybrid budget was chosen

Simulated on the Duck's published 25-game × 20-pass run (`example-run/benchmark.json`):

| Explorer opening | Mean public score |
|---|---|
| none (Duck alone) | 1.723 |
| every game, 24 actions | 1.387 |
| click-only games, 24 actions | **1.733** |

A failed opening costs the LLM efficiency on that level, and the metric
squares that cost. So the opening is gated to click-only games, where the
explorer is strongest (tn36 level 1 in 12 actions vs a baseline of 32; r11l in
21–34 vs 22). Expect a small gain, not a large one.

## Local dry runs (no GPU)

`localtest/` replaces the vLLM setup command with an OpenAI-compatible mock
model (`mock_llm_server.py`, random valid actions via the harness's `python`
tool). It then runs the notebook with `jupyter nbconvert --execute` against a
mirrored `/kaggle` layout and a local `arc_agi` gateway in competition mode.

| Test | Result |
|---|---|
| Commit mode (2 offline games) | pass: harness loop runs, placeholder parquet written |
| Scored rerun via gateway (25 games) | pass: scorecard closed; the hybrid opening solved tn36 L1 in 12 actions |
| Scored rerun with vLLM start failure | pass: fallback graph agent played all 25 games, scorecard 0.52 |

What these cannot test: loading Qwen3.8 on the GPU and vLLM itself. The first
Kaggle commit run is that test. Check its log for `vLLM server ready` and
`VLLM OPENAI SERVER QWEN SMOKE TEST`.

## Rebuilding the bundle

```bash
git clone https://github.com/Tufalabs/duck-harness && cd duck-harness
git checkout 7652836 && git apply ../duck-upgrades.patch
cd ARC3-Inference && uv sync   # Python 3.12
MULTIMODAL_UPSCALE=8 make kaggle-duck KAGGLE_DRY_RUN=true KAGGLE_USERNAME=<you> \
    KAGGLE_MAKE_SHARE_VERSION=true EXPERIMENTS_DIR=/tmp/build
tar --exclude=__pycache__ -czf taaf_bundle.tar.gz -C /tmp/build/*/kaggle/source-dataset .
python ../../arc-agi-3/llm/build_llm_notebook.py
```

## Honest expectations

The best documented public leaderboard result for this configuration (without
our hybrid additions) is **2.56**. Tufa's own Milestone #1 score was 1.21.
Run-to-run variance is large: a single game can swing the public mean by ±1.
A realistic range for this notebook is **about 1.5–2.8**. That is far above
the 0.27 plateau of programmatic agents, but below the ~3.6 leaders of August,
and nowhere near 8–9, which no public method has shown.

## Credits and licence

The Duck harness and TAAF are by Tufa Labs (Harold Bessis, Jeroen Cottaar,
Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel), released under the
MIT License. The embedded bundle contains their source with the changes in
`duck-upgrades.patch`.
