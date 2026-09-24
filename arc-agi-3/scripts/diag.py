import sys, logging, collections, os
"""Play one game and dump agent internals.  Usage: python scripts/diag.py ls20 3000"""
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.environ.get("ARC_AGENTS_DIR",os.path.join(ROOT,"vendor","ARC-AGI-3-Agents"))); logging.disable(logging.CRITICAL)
os.chdir(ROOT)
import arc_agi, importlib.util
from arc_agi import OperationMode
gid=sys.argv[1]; cap=int(sys.argv[2]) if len(sys.argv)>2 else 3000
agentf=sys.argv[3] if len(sys.argv)>3 else os.path.join(ROOT,'agent','my_agent.py')
spec=importlib.util.spec_from_file_location('m',agentf); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
env=arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make(gid)
m.MyAgent.MAX_ACTIONS=cap
a=m.MyAgent(card_id='x',game_id=gid,agent_name='a',ROOT_URL='x',record=False,arc_env=env,tags=[])
reasons=collections.Counter(); orig=a.do_action_request
def dar(act):
    reasons[(getattr(a,'_pending_reason','') or '').split(' ')[0]+('' if act.value else 'RESET')]+=1
    return orig(act)
a.do_action_request=dar
a.main()
print(gid,'levels',a.frames[-1].levels_completed,'nodes',len(a.nodes),'tier',a.active_tier,'mask lines',[k for k,v in a.ticker.line_hits.items() if a.ticker.mask is not None], 'maskcells', 0 if a.ticker.mask is None else int(a.ticker.mask.sum()))
print(' reasons',dict(reasons))
print(' useless',a.useless_simple,'tried',a.tried_simple)
