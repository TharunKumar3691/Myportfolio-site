"""Derive a local dry-run variant of the hybrid notebook.

Only test plumbing is changed: the bundle's vLLM setup command is replaced by
mock_setup.py, and per-game / fallback time budgets are shortened.
Usage: python make_test_notebook.py OUT.ipynb GAME_SECONDS FALLBACK_SECONDS
"""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
src = here.parent / "notebook" / "arc3-hybrid-llm.ipynb"
out, game_s, fb_s = sys.argv[1], sys.argv[2], sys.argv[3]
nb = json.loads(src.read_text())
subs = [
    ('print("hybrid: bundle extracted to", BUNDLE_DIR)',
     'print("hybrid: bundle extracted to", BUNDLE_DIR)\n'
     f'(BUNDLE_DIR / "setup_commands.json").write_text(json.dumps([\'"$PYTHON" {here / "mock_setup.py"}\']))\n'
     'print("TEST: setup replaced by mock")'),
    ('    bm.job_dir = WORKING_DIR\n',
     f'    bm.job_dir = WORKING_DIR\n    bm.solver.max_runtime_s_per_game = {game_s}\n    print("TEST: per-game runtime", {game_s})\n'),
    ('10.5 * 3600 - elapsed', f'{fb_s} + 0 * elapsed'),
]
for old, new in subs:
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and old in c["source"]:
            c["source"] = c["source"].replace(old, new)
            hits += 1
    assert hits == 1, (old, hits)
Path(out).write_text(json.dumps(nb))
print("wrote", out)
