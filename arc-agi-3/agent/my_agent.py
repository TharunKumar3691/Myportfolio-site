"""ARC-AGI-3 graph-exploration agent (training-free, CPU only, no internet).

Strategy
--------
Every level of an ARC-AGI-3 game is a deterministic-ish state machine whose
rules are unknown.  The agent explores it systematically:

1. Perception
   * The last layer of each frame is the observed 64x64 grid.
   * "Ticker" detection: step counters / energy bars change a few cells on
     every action regardless of what the action did.  Thin, small change
     blobs that recur on the same row/column across many transitions are
     learned and masked, so the state hash only reflects real game state.
   * Grid is segmented into single-colour connected components; these give
     the candidate click targets (ACTION6), ranked into salience tiers.

2. World model
   * A directed graph: node = hash of the masked grid, edge = (candidate
     action) -> next node / no-op / death.

3. Policy
   * Try an untested candidate action in the current node (best tier first).
   * Otherwise BFS through known edges to the nearest node that still has
     untested candidates and walk there (RESET is used as an edge to the
     level start when it helps).
   * Tiers are opened globally one at a time, so cheap / likely-useful
     actions are exhausted everywhere before expensive ones are tried.
   * Deadly transitions are never repeated.
   * On level-up the graph is rebuilt for the new level; interface knowledge
     (useless actions, ticker mask, productive click colours) carries over.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from collections import deque
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

log = logging.getLogger()

# Global wall-clock deadline shared by every game thread.  Kaggle kills the
# notebook at 12h; stop well before that so the gateway can write its output.
_PROCESS_START = time.time()
_DEADLINE_SECONDS = float(os.getenv("ARC_AGENT_DEADLINE_SECONDS", str(7.5 * 3600)))

SIMPLE_IDS = (1, 2, 3, 4, 5, 7)
CLICK_ID = 6
N_TIERS = 4           # click salience tiers (simple actions live in tier 0)
MAX_CLICKS_PER_NODE = 160


def _to_action(aid: int) -> GameAction:
    return GameAction.from_id(int(aid))


def _grid_of(frame: FrameData) -> Optional[np.ndarray]:
    layers = frame.frame
    if not layers:
        return None
    g = np.asarray(layers[-1], dtype=np.int16)
    if g.ndim != 2 or g.size == 0:
        return None
    return g


# --------------------------------------------------------------------------
# Connected components (4-connectivity, single colour) without scipy.
# --------------------------------------------------------------------------
def _segments(grid: np.ndarray, ignore: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = ignore.copy()
    out = []
    flat = grid.tolist()
    seen_l = seen.tolist()
    for y in range(h):
        row = seen_l[y]
        for x in range(w):
            if row[x]:
                continue
            c = flat[y][x]
            stack = [(y, x)]
            seen_l[y][x] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                if cy > 0 and not seen_l[cy - 1][cx] and flat[cy - 1][cx] == c:
                    seen_l[cy - 1][cx] = True
                    stack.append((cy - 1, cx))
                if cy < h - 1 and not seen_l[cy + 1][cx] and flat[cy + 1][cx] == c:
                    seen_l[cy + 1][cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and not seen_l[cy][cx - 1] and flat[cy][cx - 1] == c:
                    seen_l[cy][cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx < w - 1 and not seen_l[cy][cx + 1] and flat[cy][cx + 1] == c:
                    seen_l[cy][cx + 1] = True
                    stack.append((cy, cx + 1))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            my = sum(ys) / len(ys)
            mx = sum(xs) / len(xs)
            # click point = member cell closest to the centroid
            py, px = min(cells, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
            out.append({
                "color": c, "area": len(cells), "bbox": (y0, y1, x0, x1),
                "point": (py, px),
            })
    return out


class _Ticker:
    """Learns cells that change on (almost) every action: counters, timers."""

    def __init__(self) -> None:
        self.n_transitions = 0
        self.line_hits: dict[tuple, int] = {}
        self.line_actions: dict[tuple, set] = {}
        self.mask: Optional[np.ndarray] = None
        self.version = 0

    def observe(self, prev: np.ndarray, cur: np.ndarray, action_id: int) -> None:
        if prev.shape != cur.shape:
            return
        diff = prev != cur
        n = int(diff.sum())
        if n == 0:
            return
        self.n_transitions += 1
        pts = np.argwhere(diff)
        # group changed cells into small clusters (8-connectivity, cheap)
        keys = set()
        if n <= 400:
            clusters = self._clusters(pts)
            for cl in clusters:
                ys = [p[0] for p in cl]
                xs = [p[1] for p in cl]
                hgt = max(ys) - min(ys) + 1
                wid = max(xs) - min(xs) + 1
                if len(cl) > 8:
                    continue
                # a counter blob sits in a fixed thin band and slides along it
                if hgt <= 3:
                    keys.add(("r", min(ys), max(ys)))
                if wid <= 3:
                    keys.add(("c", min(xs), max(xs)))
        for k in keys:
            self.line_hits[k] = self.line_hits.get(k, 0) + 1
            self.line_actions.setdefault(k, set()).add(action_id)
        self._update_mask(cur.shape)

    @staticmethod
    def _clusters(pts: np.ndarray) -> list[list[tuple[int, int]]]:
        s = {(int(p[0]), int(p[1])) for p in pts}
        out = []
        while s:
            start = s.pop()
            stack = [start]
            cl = [start]
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (y + dy, x + dx)
                        if q in s:
                            s.remove(q)
                            stack.append(q)
                            cl.append(q)
            out.append(cl)
        return out

    def _update_mask(self, shape: tuple[int, int]) -> None:
        if self.n_transitions < 4:
            return
        lines = []
        for k, hits in self.line_hits.items():
            if hits >= 4 and hits >= 0.5 * self.n_transitions and (
                len(self.line_actions[k]) >= 2 or self.n_transitions >= 12
            ):
                lines.append(k)
        if not lines:
            return
        m = np.zeros(shape, dtype=bool)
        for kind, a, b in lines:
            if kind == "r":
                m[a:b + 1, :] = True
            else:
                m[:, a:b + 1] = True
        if self.mask is None or not np.array_equal(m, self.mask):
            self.mask = m
            self.version += 1


class _Node:
    __slots__ = ("key", "cands", "tiers", "untested", "edges", "deaths")

    def __init__(self, key: str, cands: list[tuple], tiers: list[int]) -> None:
        self.key = key
        self.cands = cands          # list of (action_id, (x, y) | None)
        self.tiers = tiers          # tier per candidate
        self.untested: list[list[int]] = [[] for _ in range(N_TIERS)]
        for i, t in enumerate(tiers):
            self.untested[t].append(i)
        self.edges: dict[int, str] = {}   # cand idx -> next node key (tested, changed)
        self.deaths: set[int] = set()

    def has_untested(self, max_tier: int) -> bool:
        for t in range(max_tier + 1):
            if self.untested[t]:
                return True
        return False

    def pop_untested(self, max_tier: int) -> Optional[int]:
        for t in range(max_tier + 1):
            if self.untested[t]:
                return self.untested[t].pop(0)
        return None


class MyAgent(Agent):
    MAX_ACTIONS = 30000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["record"] = False
        if len(args) >= 5:
            args = tuple(args[:4]) + (False,) + tuple(args[5:])
        super().__init__(*args, **kwargs)
        # One INFO line per action for ~110 threads would flood the log.
        logging.getLogger().setLevel(logging.WARNING)
        seed = int(hashlib.md5(str(self.game_id).encode()).hexdigest()[:8], 16)
        self.rng = random.Random(seed)
        self.ticker = _Ticker()
        self.useless_simple: dict[int, int] = {}   # action id -> no-op count
        self.tried_simple: dict[int, int] = {}
        self.good_colors: dict[int, int] = {}      # click colour -> productive count
        self._new_level(0)
        self.prev_grid: Optional[np.ndarray] = None
        self.prev_key: Optional[str] = None
        self.prev_cand: Optional[int] = None
        self.prev_action_id: Optional[int] = None
        self.expect_reset = True

    # ------------------------------------------------------------------ utils
    def append_frame(self, frame: FrameData) -> None:
        # The framework keeps every frame and records every action to disk.
        # With ~110 concurrent games and tens of thousands of actions each
        # that would exhaust RAM and disk, and nothing here needs the history,
        # so keep only a short tail and do not record.
        self.frames.append(frame)
        if len(self.frames) > 8:
            del self.frames[:-2]
        if frame.guid:
            self.guid = frame.guid

    @property
    def name(self) -> str:
        return f"{super().name}.graphx"

    def _new_level(self, level: int) -> None:
        self.level = level
        self.nodes: dict[str, _Node] = {}
        self.start_key: Optional[str] = None
        self.active_tier = 0
        self.plan: deque = deque()
        self.level_actions = 0

    def _masked(self, grid: np.ndarray) -> np.ndarray:
        m = self.ticker.mask
        if m is not None and m.shape == grid.shape:
            g = grid.copy()
            g[m] = -1
            return g
        return grid

    def _key(self, grid: np.ndarray) -> str:
        g = self._masked(grid)
        return hashlib.blake2b(g.astype(np.int8).tobytes(), digest_size=12).hexdigest() + str(g.shape)

    def _time_up(self) -> bool:
        return time.time() - _PROCESS_START > _DEADLINE_SECONDS

    # ------------------------------------------------------- candidate build
    def _build_node(self, key: str, grid: np.ndarray, avail: list[int]) -> _Node:
        cands: list[tuple] = []
        tiers: list[int] = []
        simple = [a for a in SIMPLE_IDS if a in avail]
        # demote simple actions that have been no-ops essentially always
        for a in simple:
            tried = self.tried_simple.get(a, 0)
            useless = self.useless_simple.get(a, 0)
            t = 0
            if tried >= 12 and useless >= 0.95 * tried:
                t = N_TIERS - 1
            cands.append((a, None))
            tiers.append(t)
        if CLICK_ID in avail:
            mask = self.ticker.mask
            ignore = mask.copy() if mask is not None and mask.shape == grid.shape else np.zeros(grid.shape, bool)
            segs = _segments(grid, ignore)
            if segs:
                vals, cnts = np.unique(grid[~ignore] if (~ignore).any() else grid, return_counts=True)
                bg = int(vals[int(np.argmax(cnts))])
                total = grid.size
                color_area = dict(zip(vals.tolist(), cnts.tolist()))
                scored = []
                for s in segs:
                    y0, y1, x0, x1 = s["bbox"]
                    area = s["area"]
                    c = s["color"]
                    if c == bg:
                        tier = 2 if area <= 64 else 3
                    elif area <= 2:
                        tier = 1
                    elif area <= 256:
                        tier = 0
                    elif area <= total * 0.2:
                        tier = 1
                    else:
                        tier = 3
                    good = self.good_colors.get(c, 0)
                    if good > 0 and tier > 0:
                        tier -= 1
                    # prefer rare colours and compact shapes inside a tier
                    rarity = color_area.get(c, 0) / total
                    score = rarity + (0 if good else 0.05) + self.rng.random() * 0.02
                    scored.append((tier, score, s))
                scored.sort(key=lambda t: (t[0], t[1]))
                for tier, _, s in scored[:MAX_CLICKS_PER_NODE]:
                    py, px = s["point"]
                    cands.append((CLICK_ID, (int(px), int(py)), s["color"]))
                    tiers.append(min(tier, N_TIERS - 1))
        # normalise cands to (aid, xy, color)
        cands = [c if len(c) == 3 else (c[0], c[1], None) for c in cands]
        return _Node(key, cands, tiers)

    def _get_node(self, key: str, grid: np.ndarray, avail: list[int]) -> _Node:
        node = self.nodes.get(key)
        if node is None:
            node = self._build_node(key, grid, avail)
            self.nodes[key] = node
        return node

    # --------------------------------------------------------------- planning
    def _bfs_to_frontier(self, src: str) -> Optional[list[int]]:
        """Shortest known path (list of cand idx) to a node with untested cands."""
        tier = self.active_tier
        prev: dict[str, tuple[str, int]] = {src: ("", -1)}
        q = deque([src])
        while q:
            k = q.popleft()
            node = self.nodes.get(k)
            if node is None:
                continue
            if k != src and node.has_untested(tier):
                path = []
                while k != src:
                    pk, ci = prev[k]
                    path.append(ci)
                    k = pk
                path.reverse()
                return path
            for ci, nk in node.edges.items():
                if nk not in prev and nk in self.nodes:
                    prev[nk] = (k, ci)
                    q.append(nk)
        return None

    def _any_frontier(self, tier: int) -> bool:
        return any(n.has_untested(tier) for n in self.nodes.values())

    # --------------------------------------------------------------- API
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        return getattr(self, "_give_up", False) or self._time_up()

    def _action(self, aid: int, xy: Optional[tuple], why: str) -> GameAction:
        a = _to_action(aid)
        if a.is_complex():
            x, y = xy if xy is not None else (32, 32)
            self._pending_data = {"x": int(x), "y": int(y)}
        else:
            self._pending_data = {}
        self._pending_reason = why
        self.prev_action_id = aid
        return a

    def do_action_request(self, action: GameAction) -> FrameData:
        # GameAction members are process-wide singletons shared by every game
        # thread, so never stash per-call data on them: pass it explicitly.
        data = dict(getattr(self, "_pending_data", {}) or {})
        if action.is_complex() and not data:
            data = {"x": 32, "y": 32}
        if not action.is_complex():
            data = {}
        reasoning = {"text": str(getattr(self, "_pending_reason", ""))}
        self._pending_data = {}
        raw = self.arc_env.step(action, data=data, reasoning=reasoning)
        return self._convert_raw_frame_data(raw)

    def _reset(self, why: str) -> GameAction:
        self.prev_key = None
        self.prev_cand = None
        self.prev_grid = None
        self.plan.clear()
        self.expect_reset = True
        self._pending_data = {}
        self._pending_reason = why
        self.prev_action_id = 0
        return GameAction.RESET

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        try:
            frame = super().take_action(action)
            self._net_failures = 0
            return frame
        except Exception as e:  # gateway hiccup: back off, give up if persistent
            self._net_failures = getattr(self, "_net_failures", 0) + 1
            log.warning(f"{self.game_id}: action failed ({self._net_failures}): {e}")
            if self._net_failures >= 20:
                self._give_up = True
            time.sleep(min(5.0, 0.5 * self._net_failures))
            return None

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            return self._choose(latest_frame)
        except Exception as e:  # never let a bug in the policy kill the game
            log.warning(f"{self.game_id}: policy error {type(e).__name__}: {e}")
            self.plan.clear()
            self.prev_key = None
            self.prev_cand = None
            self.prev_grid = None
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return self._reset("error -> reset")
            avail = [int(getattr(a, "value", a)) for a in (latest_frame.available_actions or [])]
            avail = [a for a in avail if a != 0] or [1]
            aid = self.rng.choice(avail)
            return self._action(aid, (self.rng.randrange(64), self.rng.randrange(64)), "error fallback")

    def _choose(self, latest_frame: FrameData) -> GameAction:
        state = latest_frame.state
        if state in (GameState.NOT_PLAYED,) or latest_frame.frame is None or len(latest_frame.frame) == 0:
            return self._reset("start")

        grid = _grid_of(latest_frame)
        if grid is None:
            return self._reset("empty frame")
        avail = [int(a) if not isinstance(a, GameAction) else a.value for a in (latest_frame.available_actions or [])]
        avail = [a for a in avail if a != 0] or [1, 2, 3, 4, 5, 6]

        # ---- level change bookkeeping
        lvl = int(latest_frame.levels_completed or 0)
        level_up = lvl > self.level
        if level_up:
            if self.prev_key is not None and self.prev_cand is not None:
                pn = self.nodes.get(self.prev_key)
                if pn is not None:
                    c = pn.cands[self.prev_cand]
                    if c[0] == CLICK_ID and c[2] is not None:
                        self.good_colors[c[2]] = self.good_colors.get(c[2], 0) + 3
            self._new_level(lvl)
            self.prev_key = None
            self.prev_cand = None
            self.prev_grid = None

        # ---- update ticker mask with the last transition
        if self.prev_grid is not None and self.prev_action_id is not None and not level_up:
            self.ticker.observe(self.prev_grid, grid, self.prev_action_id)

        key = self._key(grid)

        # ---- record the outcome of the previous action
        if self.prev_key is not None and self.prev_cand is not None:
            pn = self.nodes.get(self.prev_key)
            if pn is not None:
                aid, _, color = pn.cands[self.prev_cand]
                if state is GameState.GAME_OVER:
                    # keep any known successor: deaths are often caused by a
                    # hidden budget (energy / moves) rather than the edge itself
                    pn.deaths.add(self.prev_cand)
                else:
                    changed = key != self.prev_key
                    if aid != CLICK_ID:
                        self.tried_simple[aid] = self.tried_simple.get(aid, 0) + 1
                        if not changed:
                            self.useless_simple[aid] = self.useless_simple.get(aid, 0) + 1
                    elif changed and color is not None:
                        self.good_colors[color] = self.good_colors.get(color, 0) + 1
                    if changed:
                        pn.edges[self.prev_cand] = key
                    else:
                        pn.edges.pop(self.prev_cand, None)

        if state is GameState.GAME_OVER:
            return self._reset("game over -> reset")

        # ticker mask changed -> old hashes are stale; rebuild graph lazily
        if getattr(self, "_mask_version", 0) != self.ticker.version:
            self._mask_version = self.ticker.version
            self.nodes = {}
            self.start_key = None
            self.plan.clear()
            key = self._key(grid)

        node = self._get_node(key, grid, avail)
        if self.start_key is None:
            self.start_key = key
        self.expect_reset = False

        self.prev_grid = grid
        self.prev_key = key
        self.level_actions += 1

        # ---- follow an existing plan if we are on track
        if self.plan:
            exp_key, ci = self.plan[0]
            if exp_key == key:
                self.plan.popleft()
                self.prev_cand = ci
                aid, xy, _ = node.cands[ci]
                return self._action(aid, xy, "plan")
            self.plan.clear()

        # ---- untested action here
        while True:
            ci = node.pop_untested(self.active_tier)
            if ci is not None:
                self.prev_cand = ci
                aid, xy, _ = node.cands[ci]
                return self._action(aid, xy, f"explore t{node.tiers[ci]}")

            path = self._bfs_to_frontier(key)
            if path:
                # store as (expected node key, cand) pairs
                k = key
                steps = []
                for ci in path:
                    steps.append((k, ci))
                    k = self.nodes[k].edges[ci]
                self.plan = deque(steps)
                exp_key, ci = self.plan.popleft()
                self.prev_cand = ci
                aid, xy, _ = node.cands[ci]
                return self._action(aid, xy, "goto frontier")

            # frontier unreachable from here but reachable from level start
            if key != self.start_key and self.start_key in self.nodes and (
                self.nodes[self.start_key].has_untested(self.active_tier)
                or self._bfs_to_frontier(self.start_key)
            ):
                return self._reset("frontier via start -> reset")

            if self.active_tier < N_TIERS - 1:
                self.active_tier += 1
                continue
            break

        # ---- everything reachable is exhausted.  The level most likely has
        # hidden state (keys carried, energy, order-dependent switches) that the
        # hash cannot see.  Forget the graph and explore again from here: new
        # paths through the same screens then get tried in new contexts.
        self.exhaust_count = getattr(self, "exhaust_count", 0) + 1
        if self.exhaust_count <= 50 and len(self.nodes) > 1:
            self.nodes = {}
            self.start_key = None
            self.active_tier = 0
            self.plan.clear()
            node = self._get_node(key, grid, avail)
            self.start_key = key
            ci = node.pop_untested(0)
            if ci is not None:
                self.prev_cand = ci
                aid, xy, _ = node.cands[ci]
                return self._action(aid, xy, "re-explore")
        ci = self.rng.randrange(len(node.cands)) if node.cands else None
        if ci is None:
            return self._reset("no candidates")
        if self.rng.random() < 0.02:
            return self._reset("exhausted -> reset")
        self.prev_cand = ci
        aid, xy, _ = node.cands[ci]
        return self._action(aid, xy, "random fallback")
