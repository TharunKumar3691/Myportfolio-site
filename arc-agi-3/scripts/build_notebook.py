"""Build the Kaggle submission notebook from agent/my_agent.py.

Follows the pattern of Kaggle's official sample submission
("ARC3 Sample Submission - Stochastic Goose") and the official
ARC-AGI-3-Kaggle-Starter:

  Cell 1  install `arc-agi` from the offline competition wheels.
  Cell 2  write the agent to /tmp/my_agent.py (kept out of /kaggle/working so
          the only output file is submission.parquet).
  Cell 3  competition rerun: wait for the gateway sidecar, copy the agents
          framework, register MyAgent, run `python main.py --agent myagent`.
          The gateway records every action and writes submission.parquet.
  Cell 4  commit run (Save & Run All): write a placeholder submission.parquet
          so the commit succeeds.

Usage:  python scripts/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# The agent is pure numpy on CPU, so no accelerator is needed.
ACCELERATOR = "cpu"
_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


INTRO = """\
# ARC-AGI-3 — Graph-Exploration Agent (training-free, CPU)

An agent that learns every unseen game online, from frames alone:

* **Perception**: last frame layer → 64×64 grid. Step counters / energy bars are
  detected (thin change-blobs that recur in the same row/column band on every
  action) and masked out of the state hash. The grid is segmented into
  single-colour connected components, which become click (ACTION6) candidates
  ranked into salience tiers.
* **World model**: directed graph, node = hash of the masked grid,
  edge = candidate action → next state / no-op / death.
* **Policy**: take an untested candidate here (best tier first); otherwise BFS
  through known edges to the nearest node with untested candidates; RESET when
  the frontier is only reachable from the level start; open lower-salience
  tiers only after the higher ones are exhausted everywhere; re-explore when
  the reachable graph is exhausted (hidden state).
* **Across levels**: interface knowledge (useless keys, productive click
  colours, ticker mask) carries over; the graph is rebuilt per level.

The whole agent is in the next-but-one cell. It is thread-safe: the framework
plays every game in its own thread, and GameAction enum members are shared
singletons, so click coordinates are passed explicitly instead of being
stored on the enum.
"""


def build() -> dict:
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        f"    {COMP_DIR}/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    run_cell = code_cell(
        dedent(
            f"""\
            import os

            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Wait for the gateway sidecar to be ready.
                !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                      --retry-max-time 600 http://gateway:8001/api/games > /dev/null

                # Copy the framework into a writable location.
                !rm -rf /kaggle/working/ARC-AGI-3-Agents
                !cp -r {COMP_DIR}/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents

                # Drop our agent in as a framework template.
                !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

                # Register MyAgent. The upstream __init__.py eagerly imports LLM
                # templates whose dependencies are not installed offline.
                with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                    f.write(\"\"\"from typing import Type
            from dotenv import load_dotenv
            from .agent import Agent, Playback
            from .swarm import Swarm
            from .templates.random_agent import Random
            from .templates.my_agent import MyAgent

            load_dotenv()

            AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
                'random': Random,
                'myagent': MyAgent,
            }}
            \"\"\")

                # Point the framework at the gateway sidecar.
                with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                    f.write(\"\"\"SCHEME=http
            HOST=gateway
            PORT=8001
            ARC_API_KEY=test-key-123
            ARC_BASE_URL=http://gateway:8001/
            OPERATION_MODE=online
            ENVIRONMENTS_DIR=
            RECORDINGS_DIR=/kaggle/working/server_recording
            \"\"\")

                # Stop exploring after 7.5h so the run finishes well inside
                # Kaggle's 12h limit (all games play concurrently).
                os.environ['ARC_AGENT_DEADLINE_SECONDS'] = str(int(7.5 * 3600))

                # Run it. The gateway records every action and emits
                # submission.parquet. Logging is kept at WARNING so 110 threads
                # do not flood the notebook log.
                !cd /kaggle/working/ARC-AGI-3-Agents && \\
                    MPLBACKEND=agg ARC_AGENT_QUIET=1 \\
                    python main.py --agent myagent 2>&1 | grep -v " | INFO | " | tail -n 400
            """
        )
    )

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save & Run All (commit) mode: emit a placeholder submission so
                # the commit succeeds. The real submission.parquet is produced by
                # the gateway during the competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                print(submission.head())
            """
        )
    )

    accel = _ACCELERATORS[ACCELERATOR]
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(INTRO),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)} (accelerator: {ACCELERATOR})")
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        meta["enable_gpu"] = _ACCELERATORS[ACCELERATOR]["gpu"]
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
