"""Stand-in for the bundle's vLLM setup command in local dry runs.

Starts mock_llm_server.py and persists the same analyzer env keys that the
real setup script writes (see duck-upgrades.patch / inference/framework/kaggle.py).
Set MOCK_SETUP_FAIL=1 to simulate a vLLM start failure (tests the fallback).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if os.environ.get("MOCK_SETUP_FAIL") == "1":
    print("mock setup: simulating vLLM failure", flush=True)
    sys.exit(1)
here = Path(__file__).resolve().parent
subprocess.Popen([sys.executable, str(here / "mock_llm_server.py"), "1234"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
for _ in range(50):
    try:
        urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(0.2)
url = "http://127.0.0.1:1234/v1"
env = {
    "LOCAL_ANALYZER_BASE_URL": url, "OPENAI_BASE_URL": url,
    "LOCAL_ANALYZER_PROVIDER": "vllm", "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-27B-FP8", "INFERENCE_ANALYZER_MODEL": "Qwen/Qwen3.8-27B-FP8",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Kaggle Harness", "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0", "LOCAL_ANALYZER_TOOL_STEPS": "0", "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024", "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6", "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "1", "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "8",
}
p = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
cur = json.loads(p.read_text()) if p.exists() else {}
cur.update(env)
p.write_text(json.dumps(cur, indent=2))
print("mock setup: mock LLM server ready at", url, flush=True)
