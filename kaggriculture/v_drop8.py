"""Kaggriculture agent.

Architecture
------------
1. Forecaster  - projects end-of-season market inventory for every product from
                 remaining town demand (known shops + expected future shops) and the
                 remaining output of both farms (the opponent farm is public).
                 Marginal value of a unit = price at the projected inventory.
2. Planner     - greedily assigns free tiles to the crop / animal with the best
                 net value per tile-day, with diminishing returns as each commitment
                 is added to the projected supply. Buys seeds, animals and land.
3. Scheduler   - turns tile states into valued tasks (water, feed, care, harvest,
                 fertilize, plant, build, place, dig) and matches farmer / hands to
                 tasks every turn, handling supply pickups (wheat, fertilizer, animals).
4. Market      - sells when the current price beats the projected future price,
                 keeps feed / fertilizer reserves, hires hands for the workload.
"""
import math

# --------------------------------------------------------------------------- rules
CROPS = {
    "WHEAT":      {"seed": 10, "fyd": 2, "myd": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "fyd": 2, "myd": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "fyd": 8, "myd": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "fyd": 10, "myd": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "fyd": 10, "myd": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "fyd": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "fyd": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "fyd": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MP = {
    "WHEAT":      (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT":     (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO":     (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON":      (250, 300, "log", 0.20, "sq", 3.60),
    "EGG":        (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK":       (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL":       (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
I0 = 10000
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
TPD = 24
LAST_STEP = 718          # last step whose actions are processed (day 29, hour 22)
LAST_DAY = 29
LAST_PROD_DAY = 28       # last end-of-day refresh happens at the end of day 28
SHED_CAP = 100
SHED_ACCESS = [(4, 4), (5, 4), (4, 5), (5, 5)]
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# ------------------------------------------------------------------ tunable params
P = {
    "future_shop_weight": 0.75,  # confidence in expected (not yet unlocked) shop demand
    "opp_weight": 1.0,           # weight on opponent projected supply
    "hold_margin": 0.06,         # hold stock only if projected price beats now by this
    "labor_cost": 6.0,           # $ per unit action in valuations
    "max_hands": 16,
    "actions_per_unit": 19.0,    # usable task actions per unit per day (rest = walking)
    "min_rate": 4.0,             # minimum $ / tile-day to commit a tile
    "cash_reserve": 60,
    "fert_internal": True,
    "mirror": 0.8,               # opponent mirrors this share of our new commitments
    "mirror_days": 16.0,         # ... fading out by this day
    "tile_shadow": 12.0,         # $ per tile-day opportunity cost in planner scoring
    "hold": False,
    "melon_cap": 99,
    "melon_cap_until": 10,
    "alpha": 0.4,
    "plan_task_frac": 0.35,
    "fert_internal_value": 60.0,
    "build_value": 160.0,
    "place_value": 320.0,
    "pickup_min": 4,
    "drop_goods": 8,
    "drop_weight": 0.3,
    "eod_load": 85,
    "eod_hour": 12,
    "drop_risk_mult": 1.0,
    "urgency": 2.0,
    "work_mult": 1.4,
    "hire_base": 14,
    "opening": True,
    "hire_frac": 0.05,
    "land_fill": 900,
    "deliver_min_price": 30,
    "deliver_days": 12,
    "deliver_min_units": 3,
    "deliver_weight": 0.25,
    "land_free_max": 2,
    "land_margin": 250,
    "hold_cap": 25,
    "disc": 0.04,                # daily discount rate on day 0 (reinvestment value of cash)
    "disc_until": 14.0,          # discounting fades out by this day
}

_fib = [1, 1]
for _ in range(40):
    _fib.append(_fib[-1] + _fib[-2])


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


_AMP = {}
for _k, (_b, _T, _bf, _bt, _af, _at) in MP.items():
    _AMP[_k] = (_bt * _b / _shape(_bf, _T, _T), _at * _b / _shape(_af, _T, _T))


def price(item, inv):
    b, T, bf, bt, af, at = MP[item]
    lo, hi = _AMP[item]
    if inv < I0:
        p = b + lo * _shape(bf, I0 - inv, T)
    else:
        p = b - hi * _shape(af, inv - I0, T)
    return max(1, int(round(p)))


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def quadrant(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


# ------------------------------------------------------------- production models
def crop_prod_days(crop, pd):
    """End-of-day indices on which an ongoing crop produces."""
    cd = CROPS[crop]
    first = pd + cd["fyd"] - 1
    return [first + k * cd["interval"] for k in range(cd["max_yield"])]


def animal_prod_days(animal, placed):
    a = ANIMALS[animal]
    first = placed + a["fyd"] - 1
    out = []
    j = first
    while j <= LAST_PROD_DAY:
        out.append(j)
        j += a["interval"]
    return out


def animal_future_units(tile, day):
    """Projected future product units (incl. held) for an animal tile, with daily care."""
    an = tile["animal"]
    a = ANIMALS[an]
    units = tile.get("yield_units", 0)
    bank = tile.get("pending_care_bonus", 0) or 0
    prev = max(day, tile["placed_day"])
    for j in animal_prod_days(an, tile["placed_day"]):
        if j < day:
            continue
        units += min(a["max_held"], 1 + bank + (j - prev))
        bank = 0
        prev = j
    return units


def crop_future_units(tile, day, fert=False):
    crop = tile["crop"]
    cd = CROPS[crop]
    pd = tile["planted_day"]
    units = tile.get("yield_units", 0)
    if not cd["ongoing"]:
        age = day - pd
        ws = (cd["myd"] + 1) // 2
        u = units
        for a in range(max(age, ws), cd["myd"] + 1):
            if a == age and tile.get("watered_today"):
                continue
            if pd + a > LAST_DAY:
                break
            u += 2 if fert else 1
        u = min(cd["max_yield"], u)
        if pd + cd["fyd"] > LAST_DAY:
            return 0
        return u
    for j in crop_prod_days(crop, pd):
        if day <= j <= LAST_PROD_DAY:
            units += 2 if fert else 1
    return min(units, 2 * cd["max_yield"])


# ------------------------------------------------------------------------ memory
MEM = {}


def reset_mem():
    MEM.clear()
    MEM.update({
        "plan": {},            # (x,y) -> option name
        "plan_val": {},        # (x,y) -> expected value of the plan entry
        "targets": {},         # unit idx -> (x,y)
        "last_step": -1,
        "opp_sales": {p: 0 for p in PRODUCTS},
    })


# ------------------------------------------------------------------------ state
class S:
    pass


def build_state(obs):
    s = S()
    s.step = int(obs["step"])
    s.day = int(obs["day"])
    s.hour = int(obs["hour"])
    s.me = int(obs["player"])
    farms = obs["farms"]
    s.farm = farms[s.me]
    s.opp = farms[1 - s.me] if len(farms) > 1 else None
    s.tiles = s.farm["tiles"]
    s.money = float(s.farm["money"])
    priv = obs["private"]
    s.shed = {k: v for k, v in (priv.get("shed") or {}).items() if v}
    s.seeds = dict(priv.get("seeds") or {})
    invs = list(priv.get("inventories") or [{}])
    s.units = []
    positions = [tuple(s.farm["farmer"])] + [tuple(h) for h in s.farm.get("hands", [])]
    for i, pos in enumerate(positions):
        inv = dict(invs[i]) if i < len(invs) and invs[i] else {}
        s.units.append({"idx": i, "pos": pos, "inv": inv})
    s.market_inv = dict(obs["market"]["inventory"])
    s.prices = dict(obs["market"]["prices"])
    s.shops = list(obs["town"]["unlocked_shops"])
    s.unlocked = list(s.farm["unlocked_quadrants"])
    s.hires_today = int(s.farm.get("hires_today", 0))
    s.turns_left_today = TPD - s.hour if s.day < LAST_DAY else (LAST_STEP - s.step + 1)
    return s


# -------------------------------------------------------------------- forecaster
NDAYS = LAST_DAY + 2

EXP_SHOP_MULT = {}
for _p in PRODUCTS:
    EXP_SHOP_MULT[_p] = sum((2 if len(v) == 1 else 1) for v in SHOPS.values() if _p in v) / 8.0


def daily_demand(s):
    """Town consumption per product per day (known shops + expected future shops)."""
    dem = {p: [0.0] * NDAYS for p in PRODUCTS}
    frac_today = (TPD - s.hour) / float(TPD)
    for d in range(s.day, LAST_DAY + 1):
        f = frac_today if d == s.day else 1.0
        if d == LAST_DAY:
            f *= 23.0 / 24.0
        for p in PRODUCTS:
            if p != "FERTILIZER":
                dem[p][d] += f
        for sh in s.shops:
            prods = SHOPS[sh]
            m = 2 if len(prods) == 1 else 1
            for p in prods:
                dem[p][d] += 6 * m * f
    n_left = 8 - len(s.shops)
    k = (s.day // 3 + 1) * 3
    w = P["future_shop_weight"]
    while n_left > 0 and k <= LAST_DAY:
        for d in range(k, LAST_DAY + 1):
            for p in PRODUCTS:
                dem[p][d] += w * 6 * EXP_SHOP_MULT[p]
        n_left -= 1
        k += 3
    return dem


def tile_flows(t, day, ours):
    """Projected (day, product, units) flows of an existing farm tile."""
    out = []
    k = t.get("kind")
    if k == "PLANT":
        crop = t["crop"]
        cd = CROPS[crop]
        pd = t["planted_day"]
        if not cd["ongoing"]:
            u = crop_future_units(t, day)
            age_h = {"WHEAT": 4, "CARROT": 3, "MELON": 10}[crop]
            hd = max(day, pd + max(cd["fyd"], min(age_h, cd["myd"])))
            if hd <= LAST_DAY and u > 0:
                out.append((hd, crop, u))
        else:
            yu = t.get("yield_units", 0)
            if yu:
                out.append((day, crop, yu))
            per = 2.0 if ours else 1.6
            for j in crop_prod_days(crop, pd):
                if day <= j <= LAST_PROD_DAY:
                    out.append((j + 1, crop, per))
    elif t.get("animal"):
        an = t["animal"]
        a = ANIMALS[an]
        units = t.get("yield_units", 0)
        if units:
            out.append((day, a["product"], units))
        bank = t.get("pending_care_bonus", 0) or 0
        prev = max(day, t["placed_day"])
        eff = 1.0 if ours else 0.9
        last = None
        for j in animal_prod_days(an, t["placed_day"]):
            if j < day:
                continue
            n = min(a["max_held"], 1 + bank + (j - prev))
            out.append((j + 1, a["product"], n * eff))
            bank = 0
            prev = j
            last = j
        for d in range(day, LAST_PROD_DAY + 1):
            out.append((d + 1, "FERTILIZER", 1.0))
        if last is not None and ours:
            for d in range(day, last + 1):
                out.append((d, "WHEAT", -1.0))
    return out


def add_flows(arr, flows, weight=1.0):
    for d, p, n in flows:
        if d < NDAYS:
            arr[p][d] += n * weight


def forecast(s):
    sup = {p: [0.0] * NDAYS for p in PRODUCTS}
    for row in s.tiles:
        for t in row:
            if isinstance(t, dict):
                add_flows(sup, tile_flows(t, s.day, True))
    if s.opp is not None:
        for row in s.opp["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    fl = [(d, p, n) for d, p, n in tile_flows(t, s.day, False) if p != "WHEAT"]
                    add_flows(sup, fl, P["opp_weight"])
    # holdings are sold now
    for p, n in s.shed.items():
        if p in sup:
            sup[p][s.day] += n
    for u in s.units:
        for p, n in u["inv"].items():
            if p in sup:
                sup[p][s.day] += n
    # plan entries not yet realised
    for pos, opt in MEM["plan"].items():
        r = option_flows(s, opt, s.day)
        if r is not None:
            add_flows(sup, r[0])
    dem = daily_demand(s)
    # inventory path: level at which a unit sold on day d trades (mid-day)
    path = {}
    for p in PRODUCTS:
        inv = float(s.market_inv[p])
        arr = [inv] * NDAYS
        for d in range(s.day, NDAYS):
            arr[d] = inv + 0.5 * (sup[p][d] - dem[p][d])
            inv += sup[p][d] - dem[p][d]
        path[p] = arr
    s.path = path
    s.sup, s.dem = sup, dem
    # legacy end-of-season projection
    s.X = {p: path[p][LAST_DAY] for p in PRODUCTS}
    return s.X


def new_extra():
    return {p: [0.0] * NDAYS for p in PRODUCTS}


def add_extra(extra, flows):
    for d, p, n in flows:
        if p in extra:
            row = extra[p]
            for k in range(max(0, d), NDAYS):
                row[k] += n


def mirror(s):
    return P["mirror"] * max(0.0, 1.0 - s.day / P["mirror_days"])


def pval(s, p, d, extra=None, n=0.0):
    """Price of a unit of p sold on day d given extra committed supply."""
    d = min(max(d, s.day), LAST_DAY)
    x = s.path[p][d]
    if extra is not None:
        x += extra[p][d] * (1.0 + (mirror(s) if p not in ("WHEAT", "FERTILIZER") else 0.0))
    return price(p, int(x + n))


def mval(s, p, extra=None):
    return pval(s, p, s.day, extra)


def tval(s, p):
    """Value of a unit sold in the near term (daily task decisions)."""
    return max(1.0, float(s.prices.get(p, 1)))


# ------------------------------------------------------------------ valuations
def disc_rate(s):
    """Early cash compounds (reinvestment), so discount future cash flows early on."""
    if s.day >= P["disc_until"]:
        return 0.0
    return P["disc"] * (1.0 - s.day / P["disc_until"])


def option_flows(s, opt, d0):
    """Cash-flow skeleton of committing a tile to `opt` today.

    Returns (flows, occ, cost, fert_use, actions) where flows is a list of
    (day, product, units) and fert_use counts fertilizer consumed (<0 = produced).
    """
    if opt in CROPS:
        cd = CROPS[opt]
        rem = LAST_DAY - d0
        if opt == "MELON":
            if rem < 10:
                return None
            return [(d0 + 10, "MELON", 6)], 10, cd["seed"], 0, 16
        if opt in ("STRAWBERRY", "TOMATO"):
            pds = [j for j in crop_prod_days(opt, d0) if j <= LAST_PROD_DAY]
            if not pds:
                return None
            per_f = 2 if opt == "STRAWBERRY" else 3
            fert = (len(pds) + per_f - 1) // per_f
            occ = min(rem + 1, pds[-1] - d0 + 2)
            acts = occ * 0.7 + len(pds) + fert + 3
            return [(j + 1, opt, 2) for j in pds], occ, cd["seed"], fert, acts
        if opt == "WHEAT":
            if rem < 2:
                return None
            if rem >= 4:
                return [(d0 + 4, "WHEAT", 4)], 4, cd["seed"], 0, 7
            return [(d0 + rem, "WHEAT", rem)], rem, cd["seed"], 0, rem + 3
        if opt == "CARROT":
            if rem < 2:
                return None
            if rem >= 3:
                return [(d0 + 3, "CARROT", 3)], 3, cd["seed"], 0, 6
            return [(d0 + 2, "CARROT", 2)], 2, cd["seed"], 0, 5
    if opt in ANIMALS:
        a = ANIMALS[opt]
        placed = d0 if s.hour < 16 else d0 + 1
        pdays = animal_prod_days(opt, placed)
        if not pdays:
            return None
        flows = []
        prev = placed
        for i, j in enumerate(pdays):
            bank = j - prev if i == 0 else a["interval"]
            flows.append((j + 1, a["product"], min(a["max_held"], 1 + bank)))
            prev = j
        for t in range(placed, LAST_PROD_DAY + 1):
            flows.append((t + 1, "FERTILIZER", 1))
        for t in range(placed, pdays[-1] + 1):
            flows.append((t, "WHEAT", -1))
        days = pdays[-1] - placed + 1
        acts = days * 3.2 + len(pdays) + 3
        return flows, max(1, LAST_DAY - d0 + 1), a["cost"], -(LAST_PROD_DAY - placed + 1), acts
    return None


def option_eval(s, opt, d0, extra):
    """Return (net_value, occupancy_days, flows, cash_cost, fert_net, actions)."""
    r = option_flows(s, opt, d0)
    if r is None:
        return None
    flows, occ, cost, fert_use, acts = r
    rate = disc_rate(s)
    value = -cost - acts * P["labor_cost"]
    crop_units = 0.0
    crop_val = 0.0
    for day, p, n in flows:
        w = 1.0 / ((1.0 + rate) ** max(0, day - s.day))
        if p == "WHEAT" and n < 0:
            v = max(pval(s, "WHEAT", day, extra), 1)
        elif p == "FERTILIZER":
            v = pval(s, "FERTILIZER", day, extra) * 0.85
        else:
            v = pval(s, p, day, extra, n * 0.5)
        value += w * n * v
        if p == opt:
            crop_units += n
            crop_val += w * n * v
    if fert_use > 0 and opt in CROPS:
        fert_v = max(1.0, pval(s, "FERTILIZER", d0, extra))
        # fertilizer doubles ongoing-crop output; drop it if it does not pay
        if fert_use * fert_v > crop_val / 2.0:
            value -= crop_val / 2.0
            flows = [(d, p, n / 2.0 if p == opt else n) for d, p, n in flows]
            fert_use = 0
        else:
            value -= fert_use * fert_v
            flows = flows + [(d0, "FERTILIZER", -fert_use)]
    return value, occ, flows, cost, -fert_use, acts


OPTIONS = ["MELON", "STRAWBERRY", "TOMATO", "WHEAT", "CARROT", "SHEEP", "COW", "GOOSE"]


# ------------------------------------------------------------------- tile helpers
def tile_at(s, pos):
    return s.tiles[pos[1]][pos[0]]


def unlocked_positions(s):
    out = []
    for y in range(10):
        for x in range(10):
            if s.tiles[y][x] != "LOCKED":
                out.append((x, y))
    return out


def shed_dist(pos):
    return min(dist(pos, a) for a in SHED_ACCESS)


def nearest_access(pos):
    return min(SHED_ACCESS, key=lambda a: (dist(pos, a), SHED_ACCESS.index(a)))


# ---------------------------------------------------------------------- planner
def clean_plan(s):
    plan = MEM["plan"]
    for pos in list(plan.keys()):
        t = tile_at(s, pos)
        opt = plan[pos]
        if t == "LOCKED":
            del plan[pos]
            continue
        if isinstance(t, dict):
            k = t.get("kind")
            if k == "PLANT":
                del plan[pos]
            elif k in ("COOP", "PASTURE"):
                if t.get("animal") or opt not in ANIMALS or ANIMALS[opt]["structure"] != k:
                    del plan[pos]


def free_tiles(s):
    out = []
    for pos in unlocked_positions(s):
        if pos in MEM["plan"]:
            continue
        t = tile_at(s, pos)
        if t is None or (isinstance(t, dict) and t.get("kind") == "WEED"):
            out.append(pos)
        elif isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE") and not t.get("animal"):
            out.append(pos)
    return out


OPENING = [("COW", 2), ("SHEEP", 3), ("MELON", 5), ("WHEAT", 8)]


def opening_plan(s):
    free = sorted(free_tiles(s), key=lambda p: (shed_dist(p), p))
    i = 0
    for opt, n in OPENING:
        for _ in range(n):
            if i < len(free):
                MEM["plan"][free[i]] = opt
                i += 1


def adopt_orphans(s):
    """Animals sitting in shed/inventories without a plan entry get a tile."""
    have = {}
    for an in ANIMALS:
        have[an] = s.shed.get(an, 0) + sum(u["inv"].get(an, 0) for u in s.units)
    for pos, opt in MEM["plan"].items():
        if opt in have:
            have[opt] -= 1
    free = None
    for an, n in have.items():
        while n > 0:
            if free is None:
                free = sorted(free_tiles(s), key=lambda p: shed_dist(p))
            struct = ANIMALS[an]["structure"]
            cand = [p for p in free if isinstance(tile_at(s, p), dict) and tile_at(s, p).get("kind") == struct]
            cand = cand or [p for p in free if tile_at(s, p) is None] or free
            if not cand:
                return
            pos = cand[0]
            free.remove(pos)
            MEM["plan"][pos] = an
            n -= 1


def run_planner(s, orders, budget):
    if s.step == 0 and P["opening"]:
        opening_plan(s)
        return budget
    clean_plan(s)
    adopt_orphans(s)
    extra = new_extra()
    # animals already bought (in shed / carried) awaiting structures count as committed
    free = free_tiles(s)
    if s.day >= LAST_DAY or (s.day == LAST_DAY - 1 and s.hour > 20):
        return budget
    buys_seed = {}
    buys_animal = {}
    wheat_p = max(20, s.prices["WHEAT"])
    n_anim = count_animals(s) + sum(1 for o in MEM["plan"].values() if o in ANIMALS)
    budget -= n_anim * wheat_p * 2
    free.sort(key=lambda p: shed_dist(p))
    guard = 0
    while free and guard < 100:
        guard += 1
        best = None
        n_melon = count_crop(s, "MELON") + sum(1 for o in MEM["plan"].values() if o == "MELON")
        for opt in OPTIONS:
            if opt == "MELON" and s.day < P["melon_cap_until"] and n_melon >= P["melon_cap"]:
                continue
            r = option_eval(s, opt, s.day, extra)
            if r is None:
                continue
            value, occ, units, cost, fert, acts = r
            if value <= 0:
                continue
            if value / occ < P["min_rate"]:
                continue
            rate = value / (cost + occ * P["tile_shadow"])
            if best is None or rate > best[0]:
                best = (rate, opt, r)
        if best is None:
            break
        rate, opt, r = best
        cost = r[3]
        if cost + (wheat_p * 2 if opt in ANIMALS else 0) > budget:
            # try the best affordable option instead
            alt = None
            for opt2 in OPTIONS:
                if opt2 == "MELON" and s.day < P["melon_cap_until"] and n_melon >= P["melon_cap"]:
                    continue
                r2 = option_eval(s, opt2, s.day, extra)
                if r2 is None or r2[0] <= 0 or r2[3] + (wheat_p * 2 if opt2 in ANIMALS else 0) > budget:
                    continue
                if r2[0] / r2[1] < P["min_rate"]:
                    continue
                rate2 = r2[0] / (r2[3] + r2[1] * P["tile_shadow"])
                if rate2 < 0.5 * rate:
                    continue
                if alt is None or rate2 > alt[0]:
                    alt = (rate2, opt2, r2)
            if alt is None:
                break
            rate, opt, r = alt
            cost = r[3]
        # choose tile: animals nearest the shed, crops next
        if opt in ANIMALS:
            struct = ANIMALS[opt]["structure"]
            cand = [p for p in free if isinstance(tile_at(s, p), dict) and tile_at(s, p).get("kind") == struct]
            pos = cand[0] if cand else free[0]
        else:
            cand = [p for p in free if tile_at(s, p) is None or tile_at(s, p).get("kind") == "WEED"]
            if not cand:
                break
            pos = cand[0]
        free.remove(pos)
        MEM["plan"][pos] = opt
        MEM["plan_val"][pos] = max(0.0, r[0])
        budget -= cost
        if opt in ANIMALS:
            budget -= wheat_p * 2
        add_extra(extra, r[2])
        if opt in CROPS:
            buys_seed[opt] = buys_seed.get(opt, 0) + 1
        else:
            buys_animal[opt] = buys_animal.get(opt, 0) + 1
    return budget


def purchase_orders(s, budget):
    """Buy seeds / animals for plan entries lacking them."""
    orders = []
    need_seed = {}
    need_animal = {}
    for pos, opt in MEM["plan"].items():
        if opt in CROPS:
            need_seed[opt] = need_seed.get(opt, 0) + 1
        else:
            need_animal[opt] = need_animal.get(opt, 0) + 1
    spent = 0
    for an, n in need_animal.items():
        have = s.shed.get(an, 0) + sum(u["inv"].get(an, 0) for u in s.units)
        k = n - have
        if k > 0:
            orders.append(["BUY_ANIMAL", an, k])
            spent += k * ANIMALS[an]["cost"]
    for c, n in need_seed.items():
        k = n - s.seeds.get(c, 0)
        if k > 0:
            orders.append(["BUY_SEED", c, k])
            spent += k * CROPS[c]["seed"]
    return orders, spent


# -------------------------------------------------------------------- task model
class Task:
    __slots__ = ("op", "val", "need", "arg")

    def __init__(self, op, val, need=None, arg=None):
        self.op, self.val, self.need, self.arg = op, val, need, arg


DEADLINE_OPS = ("FEED", "CARE", "WATER", "FERTILIZE", "COLLECT_FERTILIZER")


def urgency(s):
    frac = s.hour / 23.0
    return 1.0 + P["urgency"] * frac * frac


def gen_tasks(s):
    tasks = _gen_tasks(s)
    u = urgency(s)
    for lst in tasks.values():
        for t in lst:
            if t.op in DEADLINE_OPS:
                t.val *= u
    return tasks


def _gen_tasks(s):
    tasks = {}
    day = s.day
    last_day = day >= LAST_DAY
    fert_v = max(1.0, tval(s, "FERTILIZER"))
    s.fert_use_soon = fert_reserve(s) - 2
    s.fert_demand = 0
    s.feed_demand = 0
    for pos in unlocked_positions(s):
        t = tile_at(s, pos)
        lst = []
        plan = MEM["plan"].get(pos)
        pv = MEM["plan_val"].get(pos, 0.0) * P["plan_task_frac"]
        if t is None:
            if plan in CROPS:
                if s.hour <= 21 and not last_day:
                    lst.append(Task("PLANT", max(60.0, pv), "SEED:" + plan, plan))
            elif plan in ANIMALS:
                if not last_day:
                    op = "BUILD_COOP" if ANIMALS[plan]["structure"] == "COOP" else "BUILD_PASTURE"
                    lst.append(Task(op, max(P["build_value"], pv)))
        elif isinstance(t, dict):
            k = t.get("kind")
            if k == "WEED":
                if plan is not None:
                    lst.append(Task("DIG", max(40.0, pv)))
            elif k == "PLANT":
                lst.extend(plant_tasks(s, t, fert_v))
            elif k in ("COOP", "PASTURE"):
                if t.get("animal"):
                    lst.extend(animal_tasks(s, t, fert_v))
                elif plan in ANIMALS and ANIMALS[plan]["structure"] == k and not last_day:
                    lst.append(Task("PLACE", max(P["place_value"], pv), "ANIMAL:" + plan, plan))
        if lst:
            tasks[pos] = lst
    return tasks


def plant_tasks(s, t, fert_v):
    out = []
    day = s.day
    crop = t["crop"]
    cd = CROPS[crop]
    pd = t["planted_day"]
    age = day - pd
    yu = t.get("yield_units", 0)
    m = tval(s, crop)
    watered = t.get("watered_today")
    cu = t.get("consecutive_unwatered", 0)
    fert_active = t.get("fertilized_until_day", -1) >= day
    last_day = day >= LAST_DAY
    if not cd["ongoing"]:
        ws = (cd["myd"] + 1) // 2
        in_window = ws <= age <= cd["myd"]
        future = crop_future_units(t, day, fert=fert_active)
        mature = age >= cd["fyd"]
        target = {"WHEAT": 4, "CARROT": 3, "MELON": 6}[crop]
        if crop == "WHEAT" and (fert_active or t.get("fertilized_until_day", -1) >= 0):
            target = 5
        if crop == "CARROT" and t.get("fertilized_until_day", -1) >= 0:
            target = 3
        grow_today = (not watered) and in_window and yu < cd["max_yield"]
        harvest_now = mature and (yu >= target or age >= cd["myd"] or last_day
                                  or (yu >= cd["max_yield"]))
        if harvest_now and grow_today and not last_day:
            harvest_now = False  # water first for the bonus
        if harvest_now and yu > 0:
            out.append(Task("HARVEST", 30.0 + yu * m))
            return out
        if not watered and not last_day:
            v = 0.0
            if grow_today:
                v += (2 if fert_active else 1) * m
            if cu >= 1 and pd + cd["fyd"] <= LAST_DAY:
                v += 50.0 + 0.5 * future * m
            if v > 0:
                out.append(Task("WATER", v))
        # fertilize one-time crops when fertilizer is cheap
        if (not fert_active and in_window and not last_day and yu < cd["max_yield"]
                and crop != "MELON"):
            days_left = min(cd["myd"], LAST_DAY - pd) - age + 1
            gain = min(days_left, cd["max_yield"] - yu - days_left) * m
            if gain > fert_v * 1.2 and gain > 0:
                out.append(Task("FERTILIZE", gain - fert_v, "FERTILIZER"))
                s.fert_demand += 1
        return out
    # ongoing crops
    pdays = crop_prod_days(crop, pd)
    fut = [j for j in pdays if j >= day and j <= LAST_PROD_DAY]
    prod_today = day in pdays and day <= LAST_PROD_DAY
    spent = not fut and yu == 0
    if yu > 0:
        out.append(Task("HARVEST", 15.0 + yu * m))
    if spent:
        if pd + cd["fyd"] - 1 + cd["interval"] * (cd["max_yield"] - 1) < day and not last_day:
            out.append(Task("DIG", 20.0))
        return out
    if not watered and not last_day and fut:
        v = 0.0
        if cu >= 1:
            v += 50.0 + len(fut) * m
        if prod_today and fert_active:
            v += m
        if v > 0:
            out.append(Task("WATER", v))
    # fertilize: active on day, day+1, day+2
    if not fert_active and fut and not last_day:
        cover = [j for j in fut if day <= j <= day + 2]
        if cover and cover[0] <= day + (1 if cd["interval"] >= 2 else 0):
            gain = len(cover) * m
            if gain > fert_v * 0.9:
                out.append(Task("FERTILIZE", gain, "FERTILIZER"))
                s.fert_demand += 1
    return out


def animal_tasks(s, t, fert_v):
    out = []
    day = s.day
    an = t["animal"]
    a = ANIMALS[an]
    prod = a["product"]
    m = tval(s, prod)
    pdays = animal_prod_days(an, t["placed_day"])
    fut = [j for j in pdays if j >= day]
    yu = t.get("yield_units", 0)
    last_day = day >= LAST_DAY
    if yu > 0:
        out.append(Task("HARVEST", 10.0 + yu * m))
    if t.get("fertilizer_available") and not (last_day and s.hour >= 20):
        fv = max(fert_v, P["fert_internal_value"] if s.fert_use_soon > 0 else 0.0)
        if fv >= 5.0:
            out.append(Task("COLLECT_FERTILIZER", fv))
    if not fut:
        return out
    bank = t.get("pending_care_bonus", 0) or 0
    nxt = fut[0]
    # care useful if next production happens after today and bank not capped
    care_useful = (not t.get("cared_today")) and nxt > day and (1 + bank + 1) <= a["max_held"]
    if nxt == day:
        care_useful = (not t.get("cared_today")) and len(fut) > 1
    if not t.get("fed_today"):
        v = 0.0
        if t.get("consecutive_unfed", 0) >= 1:
            v += 100.0 + a["cost"] * 0.5
        if nxt == day and bank > 0:
            v += bank * m
        # feeding is what makes today's care count (cared earlier today or still to do)
        care_counts = (nxt > day or len(fut) > 1) and (1 + bank + 1) <= a["max_held"]
        if care_useful or (t.get("cared_today") and care_counts):
            v += m
        v += 5.0
        out.append(Task("FEED", v, "WHEAT"))
        s.feed_demand += 1
    if care_useful:
        out.append(Task("CARE", m * 0.95))
    return out


# -------------------------------------------------------------------- scheduling
def unit_can(u, task, seeds_left, s):
    need = task.need
    if need is None:
        return True
    if need == "WHEAT":
        return u["inv"].get("WHEAT", 0) > 0
    if need == "FERTILIZER":
        return u["inv"].get("FERTILIZER", 0) > 0
    if need.startswith("SEED:"):
        return seeds_left.get(need[5:], 0) > 0
    if need.startswith("ANIMAL:"):
        return u["inv"].get(need[7:], 0) > 0
    return False


def shed_has(s, need):
    if need == "WHEAT":
        return s.shed.get("WHEAT", 0) > 0
    if need == "FERTILIZER":
        return s.shed.get("FERTILIZER", 0) > 0
    if need and need.startswith("ANIMAL:"):
        return s.shed.get(need[7:], 0) > 0
    return False


def step_toward(pos, target):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if dx == 0 and dy == 0:
        return "PASS"
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def schedule(s, tasks):
    """Return list of unit actions (index-aligned with s.units)."""
    n = len(s.units)
    actions = [["PASS"] for _ in range(n)]
    seeds_left = dict(s.seeds)
    done_units = set()
    claimed = set()
    carried = {"WHEAT": sum(u["inv"].get("WHEAT", 0) for u in s.units),
               "FERTILIZER": sum(u["inv"].get("FERTILIZER", 0) for u in s.units)}
    s.shed_after = dict(s.shed)
    last_day = s.day >= LAST_DAY
    turns_left = LAST_STEP - s.step

    # ---- final-day delivery: carry goods to shed before the end
    goods_units = []
    if last_day:
        for u in s.units:
            g = sum(v for k, v in u["inv"].items() if k in PRODUCTS)
            if g > 0:
                goods_units.append(u)
        for u in goods_units:
            d = shed_dist(u["pos"])
            if d + 2 >= turns_left or s.hour >= 21:
                if u["pos"] in SHED_ACCESS:
                    actions[u["idx"]] = ["DROP"]
                else:
                    actions[u["idx"]] = [step_toward(u["pos"], nearest_access(u["pos"]))]
                done_units.add(u["idx"])

    # ---- 0. deliver surplus fertilizer to the shed so it can be sold now
    s.placed = {}
    fert_needed = s.fert_demand
    fert_price = s.prices.get("FERTILIZER", 0)
    for u in s.units:
        if u["idx"] in done_units:
            continue
        f = u["inv"].get("FERTILIZER", 0)
        surplus = f - max(0, fert_needed)
        if u["pos"] in SHED_ACCESS and surplus > 0 and fert_price >= P["deliver_min_price"]:
            room = SHED_CAP - sum(s.shed.values()) - sum(s.placed.values())
            k = min(surplus, room)
            if k > 0:
                actions[u["idx"]] = ["PLACE", "FERTILIZER", k]
                s.placed["FERTILIZER"] = s.placed.get("FERTILIZER", 0) + k
                done_units.add(u["idx"])
                fert_needed -= (f - k) if f - k > 0 else 0

    # ---- 1. act on the current tile if there is something worth doing
    for u in s.units:
        if u["idx"] in done_units:
            continue
        lst = tasks.get(u["pos"])
        if not lst or u["pos"] in claimed:
            continue
        doable = [t for t in lst if unit_can(u, t, seeds_left, s) and t.val > 0.5]
        if not doable:
            continue
        t = max(doable, key=lambda t: t.val)
        act = [t.op]
        if t.op == "PLANT":
            act = ["PLANT", t.arg]
            seeds_left[t.arg] -= 1
        elif t.op == "PLACE":
            act = ["PLACE", t.arg]
        actions[u["idx"]] = act
        done_units.add(u["idx"])
        claimed.add(u["pos"])
        lst.remove(t)
        if t.op == "FEED":
            carried["WHEAT"] -= 1
        # keep the tile claimed only if more tasks remain for this unit next turn
    # ---- 2. assign remaining units to target tiles
    free_units = [u for u in s.units if u["idx"] not in done_units]
    if free_units:
        cand = []
        for pos, lst in tasks.items():
            if pos in claimed:
                continue
            for u in free_units:
                v_direct = 0.0
                v_supply = 0.0
                needs = set()
                nact = 0
                for t in lst:
                    if t.val <= 0.5:
                        continue
                    if unit_can(u, t, seeds_left, s):
                        v_direct += t.val
                        nact += 1
                    elif t.need and shed_has(s, t.need):
                        v_supply += t.val
                        needs.add(t.need)
                        nact += 1
                if v_direct <= 0 and v_supply <= 0:
                    continue
                d = dist(u["pos"], pos)
                al = P["alpha"]
                sc_direct = (v_direct ** al) / (d + nact + 0.5) if v_direct > 0 else 0
                if v_supply > 0:
                    acc = nearest_access(u["pos"])
                    d2 = dist(u["pos"], acc) + 1 + dist(acc, pos) + (len(needs) - 1)
                    sc_sup = ((v_direct + v_supply) ** al) / (d2 + nact + 0.5)
                else:
                    sc_sup = 0
                if d > turns_left:
                    continue
                # stickiness
                if MEM["targets"].get(u["idx"]) == pos:
                    sc_direct *= 1.15
                    sc_sup *= 1.15
                if sc_sup > sc_direct:
                    cand.append((sc_sup, u["idx"], pos, tuple(sorted(needs))))
                else:
                    cand.append((sc_direct, u["idx"], pos, ()))
        shed_load = sum(s.shed.values())
        carried_all = sum(sum(v for k, v in u["inv"].items() if k in PRODUCTS) for u in s.units)
        load = shed_load + carried_all
        risk = load > P["eod_load"] and s.hour >= P["eod_hour"] and s.day < LAST_DAY
        for u in free_units:
            goods = sum(v for k, v in u["inv"].items() if k in PRODUCTS)
            if goods >= P["drop_goods"] or (risk and goods >= 4):
                acc = nearest_access(u["pos"])
                d = dist(u["pos"], acc)
                gv = sum(v * max(1, s.prices.get(k, 0)) for k, v in u["inv"].items() if k in PRODUCTS)
                v = P["drop_weight"] * gv + 15.0 * max(0, goods - P["drop_goods"])
                if risk:
                    v += gv * P["drop_risk_mult"]
                cand.append(((v ** P["alpha"]) / (d + 1.0), u["idx"], ("DROP", acc), ()))
        if fert_price >= P["deliver_min_price"] and s.day <= P["deliver_days"]:
            for u in free_units:
                f = u["inv"].get("FERTILIZER", 0)
                if f >= (1 if s.day <= 3 else P["deliver_min_units"]):
                    acc = nearest_access(u["pos"])
                    d = dist(u["pos"], acc)
                    v = P["deliver_weight"] * f * fert_price
                    cand.append(((v ** P["alpha"]) / (d + 1.0), u["idx"], ("SHED", acc), ()))
        cand.sort(key=lambda c: -c[0])
        assigned = {}
        used_pos = set()
        for sc, ui, pos, needs in cand:
            if ui in assigned or pos in used_pos:
                continue
            assigned[ui] = (pos, needs)
            used_pos.add(pos)
        for u in free_units:
            ui = u["idx"]
            if ui not in assigned:
                MEM["targets"].pop(ui, None)
                # idle: go stand near shed / pick up wheat for later
                act = idle_action(s, u, carried)
                actions[ui] = act
                continue
            pos, needs = assigned[ui]
            MEM["targets"][ui] = pos
            if pos[0] == "DROP":
                acc = pos[1]
                if u["pos"] == acc:
                    room = SHED_CAP - sum(s.shed.values()) - sum(s.placed.values())
                    goods = sum(u["inv"].values())
                    if goods <= room:
                        actions[ui] = ["DROP"]
                        for k, v in u["inv"].items():
                            s.placed[k] = s.placed.get(k, 0) + v
                    else:
                        # place the most valuable product that fits
                        best = max((k for k in u["inv"] if k in PRODUCTS), key=lambda k: s.prices.get(k, 0), default=None)
                        if best and room > 0:
                            k = min(room, u["inv"][best])
                            actions[ui] = ["PLACE", best, k]
                            s.placed[best] = s.placed.get(best, 0) + k
                        else:
                            actions[ui] = ["PASS"]
                else:
                    actions[ui] = [step_toward(u["pos"], acc)]
                continue
            if pos[0] == "SHED":
                acc = pos[1]
                if u["pos"] == acc:
                    k = u["inv"].get("FERTILIZER", 0)
                    actions[ui] = ["PLACE", "FERTILIZER", k]
                    s.placed["FERTILIZER"] = s.placed.get("FERTILIZER", 0) + k
                else:
                    actions[ui] = [step_toward(u["pos"], acc)]
                continue
            if needs:
                if u["pos"] in SHED_ACCESS:
                    need = needs[0]
                    actions[ui] = pickup_action(s, u, need, carried, tasks)
                else:
                    actions[ui] = [step_toward(u["pos"], nearest_access(u["pos"]))]
            else:
                actions[ui] = [step_toward(u["pos"], pos)]
    return actions


def pickup_action(s, u, need, carried, tasks):
    if need.startswith("ANIMAL:"):
        an = need[7:]
        cnt = sum(1 for lst in tasks.values() for t in lst if t.op == "PLACE" and t.arg == an)
        have = s.shed_after.get(an, 0)
        k = max(1, min(have, cnt, 3))
        s.shed_after[an] = have - k
        return ["PICKUP", an, k]
    item = need
    have = s.shed_after.get(item, 0)
    if item == "WHEAT":
        deficit = s.feed_demand - carried["WHEAT"]
    else:
        deficit = s.fert_demand - carried["FERTILIZER"]
    share = int(math.ceil(max(deficit, 1) / max(1.0, len(s.units) / 3.0)))
    k = max(1, min(have, max(share, P["pickup_min"]), 10))
    s.shed_after[item] = have - k
    carried[item] += k
    return ["PICKUP", item, k]


def idle_action(s, u, carried):
    # at shed with idle time: nothing to do -> PASS
    return ["PASS"]


# ------------------------------------------------------------------- market side
HOLDABLE = ("STRAWBERRY", "MILK", "WOOL", "MELON")


def sell_orders(s, final=False):
    orders = []
    shed = dict(s.shed)
    s.held_total = 0
    # reserve wheat for feeding
    n_animals = 0
    for row in s.tiles:
        for t in row:
            if isinstance(t, dict) and t.get("animal"):
                n_animals += 1
    for p, n in getattr(s, "placed", {}).items():
        shed[p] = shed.get(p, 0) + n
    for p in PRODUCTS:
        have = shed.get(p, 0)
        if have <= 0:
            continue
        if final:
            orders.append(["SELL", p, have])
            continue
        if p == "WHEAT":
            keep = n_animals + 2
            have -= keep
            if have <= 0:
                continue
        if p == "FERTILIZER":
            keep = fert_reserve(s)
            have -= keep
            if have <= 0:
                continue
        k = have
        if P["hold"] and p in HOLDABLE and s.day < LAST_DAY - 1:
            future_p = price(p, int(s.X[p]))
            inv = s.market_inv[p]
            k = 0
            while k < have and price(p, inv + k) >= future_p * (1.0 - P["hold_margin"]):
                k += 1
            held_room = P["hold_cap"] - s.held_total
            if have - k > held_room:
                k = have - max(0, held_room)
            s.held_total += have - k
        if k > 0:
            orders.append(["SELL", p, k])
    return orders


def fert_reserve(s):
    # fertilizer needed by our ongoing crops over the next few days
    need = 0
    day = s.day
    for row in s.tiles:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT" and CROPS[t["crop"]]["ongoing"]:
                pds = [j for j in crop_prod_days(t["crop"], t["planted_day"]) if day <= j <= day + 4 and j <= LAST_PROD_DAY]
                if pds:
                    need += 1 if len(pds) <= 3 else 2
    for pos, opt in MEM["plan"].items():
        pass
    return need + 2


def workload(s, tasks):
    n = 0
    tiles = 0
    for pos, lst in tasks.items():
        k = sum(1 for t in lst if t.val > 3)
        if k:
            n += k
            tiles += 1
    pending = sum(1 for pos in MEM["plan"] if pos not in tasks)
    return n + tiles * 1.3 + pending * 2.5


def hire_orders(s, tasks):
    orders = []
    if s.day >= LAST_DAY and s.hour > 12:
        return orders
    w = workload(s, tasks)
    hours_left = max(1, TPD - s.hour - 1) if s.day < LAST_DAY else max(1, LAST_STEP - s.step)
    per_unit = P["actions_per_unit"] * hours_left / 23.0
    units_needed = int(math.ceil(P["work_mult"] * w / max(1.0, per_unit)))
    have = len(s.units)
    want = min(P["max_hands"] + 1, units_needed)
    k = 0
    hires = s.hires_today
    money = s.money
    while have + k < want and k < 8:
        c = _fib[hires]
        if c > money * P["hire_frac"] + P["hire_base"]:
            break
        if c > money - 5:
            break
        money -= c
        hires += 1
        k += 1
        orders.append(["HIRE"])
    return orders


def land_order(s, budget):
    n = len(s.unlocked) - 1
    if n >= 3 or s.day >= 22:
        return None
    cost = LAND_PRICES[n]
    # buy when nearly no free tiles and cash covers land + stocking it
    free = sum(1 for p in free_tiles(s))
    if free > P["land_free_max"]:
        return None
    if budget < cost + P["land_margin"] + P["land_fill"]:
        return None
    return ["BUY_LAND"]


# --------------------------------------------------------------------------- agent
def _agent(obs):
    s = build_state(obs)
    if s.step == 0 or MEM.get("last_step", 10 ** 9) > s.step:
        reset_mem()
    MEM["last_step"] = s.step
    forecast(s)
    orders = []
    # 1. tasks and unit actions (unit actions resolve before the market)
    tasks = gen_tasks(s)
    actions = schedule(s, tasks)
    # 2. sales (cash for purchases)
    sells = sell_orders(s, final=(s.day >= LAST_DAY and s.step >= LAST_STEP - 2))
    exp_income = 0.0
    for o in sells:
        inv = s.market_inv[o[1]]
        for k in range(o[2]):
            exp_income += price(o[1], inv + k)
    orders.extend(sells)
    budget = s.money + exp_income * 0.95 - P["cash_reserve"]
    # 2. plan and purchases
    budget = run_planner(s, orders, budget)
    buys, spent = purchase_orders(s, s.money + exp_income * 0.95)
    # 4. hires, feed wheat, land
    hires = hire_orders(s, tasks)
    feed_need = s.feed_demand + (count_animals(s) if s.hour >= 18 else 0)
    wheat_have = s.shed.get("WHEAT", 0) + s.placed.get("WHEAT", 0)
    if s.hour >= 18:
        wheat_have += sum(u["inv"].get("WHEAT", 0) for u in s.units)
    wheat_buy = []
    if feed_need > wheat_have and s.day < LAST_DAY:
        wheat_buy = [["BUY_PRODUCT", "WHEAT", int(feed_need - wheat_have)]]
    land = land_order(s, s.money + exp_income - spent)
    final_orders = []
    final_orders.extend(sells)
    final_orders.extend(hires)
    final_orders.extend(wheat_buy)
    if land:
        final_orders.append(land)
    final_orders.extend(buys)
    final_orders = final_orders[:10]
    return {"farmer": actions[0], "hands": actions[1:], "market": final_orders}


def count_crop(s, crop):
    return sum(1 for row in s.tiles for t in row if isinstance(t, dict) and t.get("crop") == crop)


def count_animals(s):
    return sum(1 for row in s.tiles for t in row if isinstance(t, dict) and t.get("animal"))


def agent(obs, config=None):
    try:
        return _agent(obs)
    except Exception as e:  # never crash: a crash forfeits the episode
        import traceback
        traceback.print_exc()
        return {"farmer": ["PASS"], "hands": [], "market": []}
