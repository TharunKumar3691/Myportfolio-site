"""Build the hybrid LLM submission notebook for ARC Prize 2026 - ARC-AGI-3.

The notebook runs Tufa Labs' open-source "Duck" harness (MIT, Milestone #1
winner) with three upgrades that public leaderboard experiments found to help
(Qwen3.8-27B-FP8 weights, 512x512 board images, FP8 KV cache), plus two
additions of ours:

* a graph-exploration opening for click-only games (see
  `inference/agent/graph_opening.py` in the bundle), and
* a fallback: if the vLLM server cannot be started during the scored rerun,
  the notebook plays every game with the CPU graph-exploration agent
  (`agent/my_agent.py`) instead of scoring 0.

The harness source + pickled solver/benchmark ("source bundle") is embedded in
the notebook as a base64 tarball, so the only external inputs are the public
vLLM wheelhouse and Qwen3.8 weight datasets. Rebuild the bundle with
`llm/README.md`; rebuild this notebook with:

    python llm/build_llm_notebook.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TGZ = ROOT / "llm" / "taaf_bundle.tar.gz"
GRAPH_AGENT = ROOT / "agent" / "my_agent.py"
OUT_DIR = ROOT / "llm" / "notebook"
NOTEBOOK = OUT_DIR / "arc3-hybrid-llm.ipynb"
METADATA = OUT_DIR / "kernel-metadata.json"

KAGGLE_USER = "tharunkumar369"
KERNEL_SLUG = "arc-agi-3-hybrid-llm-agent"
WHEELHOUSE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL = "jakobbrggen/qwen3-8-27b-fp8-hf-snapshot"
# Index 0 is a placeholder ref for the embedded bundle (mapped to BUNDLE_DIR).
BUNDLE_REF = f"{KAGGLE_USER}/embedded-taaf-bundle"
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [],
            "execution_count": None, "source": src}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


INTRO = """\
# ARC-AGI-3 — Hybrid LLM agent (Duck harness + Qwen3.8 + graph explorer)

**Solver.** Tufa Labs' open-source *Duck* harness (MIT; ARC-AGI-3 Milestone #1
winner, by Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal
Tesnar and Stefano Viel — https://github.com/Tufalabs/duck-harness): a local
LLM served by vLLM plays each game by writing Python in a sandbox that can
inspect the board (image + segmentation) and call `action(...)`.

**Changes in this notebook's bundle**
1. Model: **Qwen3.8-27B-FP8** (instead of Qwen3.6-27B-FP8).
2. Board images at **512×512** (`MULTIMODAL_UPSCALE=8`, was 4).
3. **FP8 KV cache** in vLLM.
4. **Hybrid opening**: on click-only games a graph explorer gets up to 24
   clicks on the first level before the LLM takes over (it solves some click
   puzzles in ~12–21 actions). Toggle with `HYBRID_GRAPH_OPENING=0`.
5. **Fallback**: if vLLM cannot start during the scored rerun, every game is
   played by a CPU graph-exploration agent instead of scoring 0.

**Inputs:** competition data, `driessmit1/arc3-vllm-h100-wheelhouse-v3`,
`jakobbrggen/qwen3-8-27b-fp8-hf-snapshot`. **Accelerator:** RTX Pro 6000.
Internet off.

**Save & Run All** (commit) only does a short smoke run (vLLM start + 2 games,
~30 min) to validate the stack and then writes a placeholder
`submission.parquet`. The full run happens in the competition rerun.
"""

CELL_ENV = """\
import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# True only inside a real competition rerun.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
# RESET keeps the current level (competition semantics).
os.environ["ONLY_RESET_LEVELS"] = "true"
# Hybrid opening for click-only games (see bundle: inference/agent/graph_opening.py).
os.environ.setdefault("HYBRID_GRAPH_OPENING", "1")
os.environ.setdefault("HYBRID_GRAPH_OPENING_BUDGET", "24")
# Commit-mode smoke run: number of offline games and wall-clock cap (minutes).
COMMIT_SMOKE_GAMES = int(os.environ.get("COMMIT_SMOKE_GAMES", "2"))
COMMIT_SMOKE_MINUTES = float(os.environ.get("COMMIT_SMOKE_MINUTES", "25"))

cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIR = Path("__COMP__")
print(f"hybrid: TRUE_SUBMISSION={TRUE_SUBMISSION}")
"""

CELL_INSTALL = """\
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
     "--disable-pip-version-check", "--find-links", str(COMP_DIR / "arc_agi_3_wheels"), "arc-agi"],
    stdout=subprocess.DEVNULL,
)
# python-dotenv is only needed by the fallback agent framework; never fatal.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
     "--disable-pip-version-check", "--find-links", str(COMP_DIR / "arc_agi_3_wheels"), "python-dotenv"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
"""

CELL_BUNDLE_HEAD = """\
# The TAAF source bundle (harness source + pickled solver/benchmark), embedded
# so no private dataset is needed. Extracted to a writable directory.
import base64, io, shutil, tarfile

BUNDLE_DIR = WORKING_DIR / "taaf_bundle"
shutil.rmtree(BUNDLE_DIR, ignore_errors=True)
BUNDLE_DIR.mkdir(parents=True)
BUNDLE_B64 = (
"""

CELL_BUNDLE_TAIL = """\
)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode="r:gz") as tar:
    tar.extractall(BUNDLE_DIR)
del BUNDLE_B64
assert (BUNDLE_DIR / "taaf-kaggle-bundle.json").is_file(), "bundle marker missing"
print("hybrid: bundle extracted to", BUNDLE_DIR)
"""

CELL_INPUTS = """\
DATASET_SOURCES = ["__BUNDLE_REF__", "__WHEELHOUSE__", "__MODEL__"]
KERNEL_SOURCES = []
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"


def _dataset_mount_candidates(ref: str) -> list:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _first_existing(candidates):
    return next((c for c in candidates if c.exists()), None)


kaggle_input_paths = {}
for i, ref in enumerate(DATASET_SOURCES):
    candidates = _dataset_mount_candidates(ref)
    resolved = BUNDLE_DIR if i == 0 else _first_existing(candidates)
    kaggle_input_paths[ref] = str(resolved or candidates[0])

setup_env = {
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\\n")
print("hybrid: input paths =", setup_env["TAAF_KAGGLE_INPUT_PATHS"])
"""

CELL_SETUP = """\
def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


def _command_env() -> dict:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
    return env


source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\\n" for entry in source_entries))

# Setup = install vLLM from the wheelhouse, start the OpenAI-compatible server
# with Qwen3.8-27B-FP8, smoke-test it. If that fails in the scored rerun we fall
# back to the CPU graph agent instead of crashing (which would score 0).
LLM_READY = False
try:
    env = _command_env()
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        print("hybrid: running setup command", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
        env = _command_env()
        os.environ.update(env)
    for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    LLM_READY = True
    print("hybrid: LLM stack ready", flush=True)
except Exception as exc:
    log_path = WORKING_DIR / "vllm-openai-server.log"
    tail = log_path.read_text(errors="replace").splitlines()[-60:] if log_path.exists() else []
    print(f"hybrid: LLM setup FAILED: {type(exc).__name__}: {exc}", flush=True)
    print("\\n".join(tail), flush=True)
    if not TRUE_SUBMISSION:
        raise  # surface the problem during Save & Run All
"""

CELL_LOAD = """\
if LLM_READY:
    with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
        target = pickle.load(file)
    target.actual_run_as_submission = TRUE_SUBMISSION
    target.is_competition_rerun = TRUE_SUBMISSION
    with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
        bm = pickle.load(file)
    bm.job_dir = WORKING_DIR
    print("hybrid: benchmark loaded:", bm.label)
"""

CELL_RUN = """\
def _competition_games():
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def _offline_games(env_dir: str):
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


def _write_placeholder_submission():
    import pandas as pd

    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
        WORKING_DIR / "submission.parquet", index=False
    )


def _run_fallback_graph_agent():
    \"\"\"Play every game with the CPU graph-exploration agent via the official framework.\"\"\"
    fw = WORKING_DIR / "ARC-AGI-3-Agents"
    shutil.rmtree(fw, ignore_errors=True)
    shutil.copytree(COMP_DIR / "ARC-AGI-3-Agents", fw)
    (fw / "agents" / "templates" / "my_agent.py").write_text(GRAPH_AGENT_SOURCE)
    (fw / "agents" / "__init__.py").write_text(
        "from typing import Type\\n"
        "from dotenv import load_dotenv\\n"
        "from .agent import Agent, Playback\\n"
        "from .swarm import Swarm\\n"
        "from .templates.random_agent import Random\\n"
        "from .templates.my_agent import MyAgent\\n\\n"
        "load_dotenv()\\n\\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}\\n"
    )
    (fw / ".env").write_text(
        "SCHEME=http\\nHOST=gateway\\nPORT=8001\\nARC_API_KEY=test-key-123\\n"
        "ARC_BASE_URL=http://gateway:8001/\\nOPERATION_MODE=online\\nENVIRONMENTS_DIR=\\n"
        f"RECORDINGS_DIR={WORKING_DIR / 'server_recording'}\\n"
    )
    elapsed = time.time() - NOTEBOOK_START_EPOCH
    fenv = os.environ.copy()
    fenv["ARC_AGENT_DEADLINE_SECONDS"] = str(int(max(600.0, 10.5 * 3600 - elapsed)))
    fenv["MPLBACKEND"] = "agg"
    print("hybrid: FALLBACK graph agent, deadline", fenv["ARC_AGENT_DEADLINE_SECONDS"], "s", flush=True)
    subprocess.run([sys.executable, "main.py", "--agent", "myagent"], cwd=fw, env=fenv, check=False)


import shutil

os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))
if TRUE_SUBMISSION:
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    _wait_for_gateway(os.environ["ARC_BASE_URL"])

if not LLM_READY:
    # Only reachable in the scored rerun (commit mode re-raises in setup).
    _run_fallback_graph_agent()
else:
    print((BUNDLE_DIR / "preamble.txt").read_text())
    (WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())
    if TRUE_SUBMISSION:
        bm.games = _competition_games()
        soft_end = None
    else:
        # Save & Run All: short smoke run on a few bundled public games.
        bm.games = _offline_games(str(COMP_DIR / "environment_files"))[:COMMIT_SMOKE_GAMES]
        soft_end = datetime.now() + timedelta(minutes=COMMIT_SMOKE_MINUTES)
    bm.n_passes = 1
    bm.game_weights = None
    print(f"hybrid: playing {len(bm.games)} games", flush=True)
    try:
        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
    finally:
        for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
            subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())

if not TRUE_SUBMISSION:
    # Commit runs are not scored; Kaggle still needs a submission.parquet output.
    # In the rerun the gateway writes the real one.
    _write_placeholder_submission()
    print("hybrid: wrote placeholder submission.parquet")
"""

CELL_DIAG = """\
diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    from html import escape
    from IPython.display import HTML, display

    display(HTML(f'<iframe srcdoc="{escape(diagnostics_html.read_text(), quote=True)}" '
                 'width="100%" height="900" style="border:0"></iframe>'))
else:
    print("No diagnostics.html (minimal diagnostics in the scored rerun).")
"""


def build() -> dict:
    b64 = base64.b64encode(BUNDLE_TGZ.read_bytes()).decode()
    lines = "".join(f'    "{b64[i:i + 120]}"\n' for i in range(0, len(b64), 120))
    bundle_cell = CELL_BUNDLE_HEAD + lines + CELL_BUNDLE_TAIL
    agent_src = GRAPH_AGENT.read_text()
    agent_cell = (
        "# CPU graph-exploration agent, used only as the fallback when vLLM cannot start.\n"
        "GRAPH_AGENT_SOURCE = " + json.dumps(agent_src) + "\n"
    )
    cells = [
        md(INTRO),
        md("## 1. Mode and configuration"),
        code(CELL_ENV.replace("__COMP__", COMP)),
        md("## 2. Install the ARC runtime (offline competition wheels)"),
        code(CELL_INSTALL),
        md("## 3. Embedded harness bundle"),
        code(bundle_cell),
        code(agent_cell),
        md("## 4. Locate attached datasets (vLLM wheelhouse, Qwen3.8 weights)"),
        code(CELL_INPUTS.replace("__BUNDLE_REF__", BUNDLE_REF).replace("__WHEELHOUSE__", WHEELHOUSE)
             .replace("__MODEL__", MODEL)),
        md("## 5. Start vLLM (falls back to the graph agent if this fails in the rerun)"),
        code(CELL_SETUP),
        md("## 6. Load the solver and benchmark"),
        code(CELL_LOAD),
        md("## 7. Play"),
        code(CELL_RUN),
        md("## 8. Diagnostics"),
        code(CELL_DIAG),
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1) + "\n")
    METADATA.write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
        "title": "ARC-AGI-3 Hybrid LLM Agent",
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [WHEELHOUSE, MODEL],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [],
        "machine_shape": "NvidiaRtxPro6000",
    }, indent=2) + "\n")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)} ({NOTEBOOK.stat().st_size // 1024} KiB) and {METADATA.name}")


if __name__ == "__main__":
    main()
