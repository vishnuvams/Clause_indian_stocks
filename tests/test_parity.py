"""Parity with the ORIGINAL bot: the original functions are extracted from the source file with `ast`
and executed next to the ports in `indian_stocks.rules`, on thousands of random inputs.

Set EMA_BOT_PATH to the original file (default below); the whole module is skipped if it is absent.
"""
import ast
import math
import os
import random
import time

import pytest

from indian_stocks import rules
from indian_stocks.frames import Frame

PATH = os.environ.get("EMA_BOT_PATH", "/home/user/clude_crypto/ema_9_21_50_stocks_crypto_1day_1hour.py")
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason=f"original bot not found at {PATH}")

FUNCS = {"_weak_run_decision", "_lower_high_stats", "_lower_high_fails", "_limit_extension_pct",
         "_near_extreme_dist"}
METHODS = {"calculate_ema", "_closed_candle_confirms", "_raw_setup", "_check_exit", "_entry_band_for",
           "_daily_ribbon_confirms"}


@pytest.fixture(scope="module")
def orig():
    tree = ast.parse(open(PATH, encoding="utf-8").read())
    ns = {"math": math, "time": time}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            exec(compile(ast.Module([node], []), PATH, "exec"), ns)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EMAPullbackTrader")
    stub = type("Stub", (), {})
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name in METHODS:
            exec(compile(ast.Module([f], []), PATH, "exec"), ns)
            setattr(stub, f.name, ns[f.name])
    return ns, stub


def rnd_closes(rng, n):
    p, out = 100.0, []
    for _ in range(n):
        p *= 1 + rng.gauss(0.001, 0.02)
        out.append(p)
    return out


def test_calculate_ema_parity(orig):
    ns, Stub = orig
    rng = random.Random(1)
    t = Stub()
    for _ in range(300):
        closes = rnd_closes(rng, rng.randint(1, 260))
        for period in (9, 21, 50):
            a, b = t.calculate_ema(closes, period), rules.ema_value(closes, period)
            assert (a is None and b is None) or abs(a - b) < 1e-9 * max(1, abs(a))


def test_frame_ema_matches_original(orig):
    """Frame.ema_done(k) == calculate_ema(closes[:k+1]);  ema_live(k, x) == calculate_ema(closes[:k+1] + [x])."""
    _, Stub = orig
    t = Stub()
    rng = random.Random(2)
    closes = rnd_closes(rng, 200)
    n = len(closes)
    F = Frame(list(range(n)), list(range(1, n + 1)), closes, closes, closes, closes, [1.0] * n)
    for k in (0, 5, 8, 9, 20, 21, 48, 49, 50, 120, 199):
        x = closes[k] * 1.013
        for p in (9, 21, 50):
            a, b = t.calculate_ema(closes[:k + 1], p), F.ema_done(p, k)
            assert (a is None and b is None) or abs(a - b) < 1e-9 * abs(a)
            a, b = t.calculate_ema(closes[:k + 1] + [x], p), F.ema_live(p, k, x)
            assert (a is None and b is None) or abs(a - b) < 1e-9 * abs(a), (k, p)


def test_weak_run_decision_parity(orig):
    ns, _ = orig
    rng = random.Random(3)
    hits = {"lower_close": 0, "time_limit": 0, "": 0}
    for _ in range(4000):
        closes = rnd_closes(rng, rng.randint(5, 80))
        period = rng.choice([3, 9, 21, 50])
        mx = rng.choice([2, 3, 4])
        a = ns["_weak_run_decision"](closes, period, True, mx)          # the original, LONG side
        b = rules.weak_run_decision(closes, period, mx)
        assert a == b
        hits[a[2]] += 1
    assert hits["lower_close"] > 50 and hits["time_limit"] > 5       # the test actually exercised both branches


def test_small_helpers_parity(orig):
    ns, _ = orig
    rng = random.Random(4)
    for _ in range(500):
        closes = rnd_closes(rng, rng.randint(1, 40))
        r, p = rng.randint(2, 8), rng.randint(2, 8)
        orig_stats = ns["_lower_high_stats"](closes, r, p)               # (recent_high, prior_high, recent_low, prior_low)
        mine = rules.lower_high_stats(closes, r, p)
        assert (orig_stats is None) == (mine is None)
        if mine:
            assert orig_stats[:2] == mine
            assert ns["_lower_high_fails"]("LONG", orig_stats) == rules.lower_high_fails(mine)
        hi, lo, last, lim = max(closes), min(closes), closes[-1], closes[-1] * rng.uniform(0.7, 1.3)
        assert ns["_limit_extension_pct"]("LONG", hi, lo, lim) == rules.limit_extension_pct(hi, lim)
        assert ns["_near_extreme_dist"]("LONG", last, hi, lo) == rules.near_extreme_dist(last, hi)


def test_closed_candle_and_raw_setup_parity(orig):
    _, Stub = orig
    rng = random.Random(5)
    t = Stub()
    t.ENABLE_SHORTS = False
    t.entry_band_label, t.entry_band_label_short = "EMA9", "EMA21"
    t._entry_band_for = lambda d, t=t: t.entry_band_label
    for _ in range(3000):
        e = [rng.choice([None, rng.uniform(90, 110)]) if rng.random() < 0.05 else rng.uniform(90, 110) for _ in range(3)]
        e9, e21, e50 = e
        assert t._raw_setup(e9, e21, e50, 100.0) == rules.raw_setup(e9, e21, e50, "EMA9")
        close = rng.uniform(90, 110)
        for band in ("EMA9", "EMA21", "EMA50"):
            assert t._closed_candle_confirms("LONG", close, e9, e21, e50, band_label=band) == \
                rules.closed_candle_confirms(close, e9, e21, e50, band)


def test_daily_ribbon_parity(orig):
    _, Stub = orig
    rng = random.Random(6)
    t = Stub()
    t.daily_ribbon_confirm = True
    t._today = lambda: "d"
    t._daily_ribbon_cache_date = "d"
    for _ in range(2000):
        trip = tuple(None if rng.random() < 0.1 else rng.uniform(90, 110) for _ in range(3))
        t._daily_ribbon_cache = trip
        assert t._daily_ribbon_confirms("LONG") == rules.daily_ribbon_confirms(*trip)


@pytest.mark.parametrize("mode", ["ema9_ema21", "ema9_ema50", "ema9_ema21_ema50"])
@pytest.mark.parametrize("ref", ["ema9", "ema21", "ema50", "off"])
def test_check_exit_parity(orig, mode, ref):
    """The whole exit ladder: the original `_check_exit` (stubbed broker) vs rules.check_exit."""
    _, Stub = orig
    rng = random.Random(hash((mode, ref)) & 0xFFFF)
    fired = 0
    for _ in range(600):
        closes = rnd_closes(rng, rng.randint(55, 130))
        F = Frame(list(range(len(closes))), list(range(1, len(closes) + 1)), closes, closes, closes, closes,
                  [1.0] * len(closes))
        k = len(closes) - 1
        e9, e21, e50 = F.triple_done(k)
        t = Stub()
        t.position_type = "LONG"
        t.entry_price = closes[-1] * rng.uniform(0.9, 1.1)
        t.EMA_SLOW, t.EMA_MID, t.EMA_FAST = 50, 21, 9
        t.exit_profit_2closes, t.exit_lower_close_max_closes, t.exit_mode = ref, 3, mode
        t.symbol = "X"
        t._display_price = lambda p: f"{p:.2f}"
        closed = []
        t._market_close_position = lambda reason, current_price, closed=closed: closed.append(reason)
        got = t._check_exit(closes[-1], e9, e21, e50, closes_done=closes)
        mine = rules.check_exit(closes, e9, e21, e50, mode, ref, 3)
        assert got == (mine is not None), (mode, ref)
        fired += got
    assert fired > 20
