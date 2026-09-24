"""Local evaluator: plays every public game with an agent file and computes
the official RHAE score (per level (baseline/actions)^2 capped at 1.15,
level-index weighted per game, averaged over games, 0-100 scale).

Setup (Python 3.12):
    pip install "arc-agi>=0.9.6" python-dotenv
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents vendor/ARC-AGI-3-Agents
    # replace vendor/ARC-AGI-3-Agents/agents/__init__.py with the slim version
    # written by the notebook (it only needs Agent, Playback, Swarm, Random)

Usage:
    python scripts/eval_local.py --cap 3000            # all games, 3000 actions each
    python scripts/eval_local.py --games ls20,vc33
"""
import argparse, importlib.util, json, logging, sys, time, os
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("ARC_AGENTS_DIR", os.path.join(ROOT, "vendor", "ARC-AGI-3-Agents")))
logging.disable(logging.CRITICAL)

def load(path):
    spec = importlib.util.spec_from_file_location("ua_"+str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.MyAgent

def play(args):
    path, gid, cap, tcap = args
    import arc_agi
    from arc_agi import OperationMode
    os.chdir(ROOT)
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(gid)
    Cls = load(path)
    Cls.MAX_ACTIONS = cap
    ag = Cls(card_id="x", game_id=gid, agent_name="a", ROOT_URL="http://x", record=False, arc_env=env, tags=[])
    # instrument per-level action counts
    level_marks = []
    orig = ag.append_frame
    def hook(f):
        if f.levels_completed > (level_marks[-1][0] if level_marks else 0):
            level_marks.append((f.levels_completed, ag.action_counter + 1))
        orig(f)
    ag.append_frame = hook
    t = time.time()
    import agents.agent as AA
    deadline = t + tcap
    orig_done = ag.is_done
    ag.is_done = lambda fr, lf: orig_done(fr, lf) or time.time() > deadline
    try:
        ag.main()
        err = None
    except Exception as e:
        import traceback; err = traceback.format_exc()[-800:]
    return gid, level_marks, ag.action_counter, time.time() - t, err

def score(gid, marks, baselines):
    b = baselines[gid]
    per = []; prev = 0
    for lv, cnt in marks:
        acts = cnt - prev; prev = cnt
        per.append(min(1.15, (b[lv-1] / max(acts, 1)) ** 2) if lv-1 < len(b) else 0)
    w = sum(range(1, len(b)+1))
    return 100 * sum((i+1) * s for i, s in enumerate(per)) / w

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(ROOT, "agent", "my_agent.py")); ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--tcap", type=float, default=600); ap.add_argument("--games", default="")
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import arc_agi
    from arc_agi import OperationMode
    os.chdir(ROOT)
    # NORMAL mode downloads the public games once into ./environment_files
    envs = arc_agi.Arcade(operation_mode=OperationMode.NORMAL).get_environments()
    for e in envs:
        arc_agi.Arcade(operation_mode=OperationMode.NORMAL).make(e.game_id.split('-')[0])
    baselines = {e.game_id.split('-')[0]: e.baseline_actions for e in envs}
    gids = [g for g in baselines if not a.games or g in a.games.split(',')]
    path = os.path.abspath(a.agent)
    with Pool(a.procs) as p:
        res = p.map(play, [(path, g, a.cap, a.tcap) for g in gids])
    tot = 0; lv = 0; rows = {}
    for gid, marks, n, dt, err in sorted(res):
        s = score(gid, marks, baselines); tot += s; lv += len(marks)
        rows[gid] = dict(levels=len(marks), marks=marks, score=s)
        print(f"{gid:6} lv={len(marks)}/{len(baselines[gid])} acts={n:6} t={dt:6.1f}s score={s:6.2f} marks={marks[:6]} base={baselines[gid][:len(marks)+1]}" + (f"\n ERR {err}" if err else ""))
    print(f"TOTAL levels={lv} mean_score={tot/len(gids):.3f}")
    if a.out: json.dump(rows, open(a.out, 'w'))
