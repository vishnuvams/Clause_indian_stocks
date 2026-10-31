"""Differential test of the whole selection pipeline: ORIGINAL `EMAPullbackBot._refresh_stocks` vs
`selection.build_watchlist` - same candles, same config -> the same ORDERED watchlist with the same RS sides."""
import collections
import contextlib
import dataclasses
import io
import random

import pytest

from indian_stocks.config import StrategyConfig
from indian_stocks.frames import Frame
from indian_stocks.selection import Candidate, build_watchlist
from tests import oracle

pytestmark = pytest.mark.skipif(not oracle.original_available(), reason="original bot not found (set EMA_BOT_PATH)")

# original attribute  <-  StrategyConfig field
MAP = {
    "min_volume_usd": "min_turnover_inr", "adr_min_pct": "adr_min_pct", "adr_period": "adr_period",
    "rs_enabled": "rs_enabled", "rs_lookback_days": "rs_lookback_days", "rs_min_pct_long": "rs_min_pct_long",
    "rs_min_pct_short": "rs_min_pct_short", "rs_top_n_long": "rs_top_n_long", "rs_top_n_short": "rs_top_n_short",
    "enable_shorts": "enable_shorts", "ema_fast": "ema_fast", "ema_mid": "ema_mid", "ema_slow": "ema_slow",
    "near_high_max_pct": "near_high_max_pct", "near_high_lookback": "near_high_lookback",
    "recent_rs_filter": "recent_rs_filter", "recent_rs_days": "recent_rs_days",
    "recent_rs_min_pct": "recent_rs_min_pct", "ema21_rising_filter": "ema21_rising_filter",
    "ema21_slope_days": "ema21_slope_days", "ema21_slope_min_pct": "ema21_slope_min_pct",
    "lower_high_filter": "lower_high_filter", "lower_high_recent_days": "lower_high_recent_days",
    "lower_high_prior_days": "lower_high_prior_days",
}
FLOOR = 4.5e7


class FakeUniverseInfo:
    def __init__(self, candles, vols):
        self.candles, self.vols = candles, vols

    def post(self, path, payload):
        assert payload["type"] == "metaAndAssetCtxs"
        syms = list(self.vols)
        return [{"universe": [{"name": f"xyz:{s}"} for s in syms]}, [{"dayNtlVlm": str(self.vols[s])} for s in syms]]

    def candles_snapshot(self, req):
        return list(self.candles.get(req["coin"], []))


def make_candles(rng, n, drift, vol, p0=100.0):
    out, p = [], p0
    for i in range(n):
        o = p * (1 + rng.gauss(0, vol / 5))
        c = o * (1 + rng.gauss(drift, vol))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        out.append({"t": i * 86_400_000, "o": o, "h": h, "l": l, "c": c, "v": 1})
        p = c
    return out


def universe(rng, n_syms=36, bench=True):
    candles, vols = {}, {}
    for i in range(n_syms):
        n = 250 if rng.random() > 0.1 else rng.randint(20, 120)        # some young stocks: short history
        drift = rng.choice([0.012, 0.008, 0.004, 0.0, -0.004, -0.008, -0.012])
        cs = make_candles(rng, n, drift, rng.choice([0.012, 0.02, 0.03]))
        if n == 250 and rng.random() < 0.12:                         # corrupt feed: zero lows => no usable ADR
            for c in cs[-30:]:
                c["l"] = 0.0
        candles[f"xyz:S{i:02d}"] = cs
        vols[f"S{i:02d}"] = FLOOR * rng.choice([0.2, 0.9, 1.2, 3, 10])
    if bench:
        candles["xyz:SP500"] = make_candles(rng, 250, 0.002, 0.008)
    return candles, vols


def original_watchlist(mod, cfg, candles, vols):
    bot = object.__new__(mod.EMAPullbackBot)
    bot.info = FakeUniverseInfo(candles, vols)
    bot.dex_name, bot.is_stocks, bot.blacklist = "xyz", True, set()
    bot.adr_fetch_sleep_sec, bot.rs_benchmark = 0, "SP500"
    bot._candle_api_params = ["req"]
    for attr, field in MAP.items():
        setattr(bot, attr, getattr(cfg, field))
    captured = []
    bot._apply_selection = lambda ordered, meta=None: captured.append(dict(ordered))
    with contextlib.redirect_stdout(io.StringIO()):
        bot._refresh_stocks()
    assert len(captured) == 1, "the original did not reach _apply_selection"
    return list(captured[0].items())


def frame_of(cs, turnover_last=0.0):
    done = cs[:-1]
    n = len(done)
    tv = [0.0] * n
    if n:
        tv[-1] = turnover_last
    return Frame(list(range(n)), list(range(1, n + 1)), [c["o"] for c in done], [c["h"] for c in done],
                 [c["l"] for c in done], [c["c"] for c in done], tv)


def my_watchlist(cfg, candles, vols):
    cands = []
    for coin, cs in candles.items():
        sym = coin.split(":")[1]
        if sym == "SP500" or sym not in vols:
            continue
        D = frame_of(cs, vols[sym])
        cands.append(Candidate(sym, D, len(D) - 1, cs[-1]["c"]))
    bc = candles.get("xyz:SP500")
    B = frame_of(bc) if bc else None
    out, dropped = build_watchlist(cands, B, (len(B) - 1) if B else -1, cfg)
    return [(s.symbol, s.direction) for s in out], dropped


def random_cfg(rng):
    return dataclasses.replace(
        StrategyConfig(),
        min_turnover_inr=FLOOR, adr_min_pct=rng.choice([0.0, 2.0, 4.0]),
        rs_enabled=rng.random() > 0.15, rs_lookback_days=rng.choice([20, 30]),
        rs_min_pct_long=rng.choice([-15.0, 0.0, 5.0, 10.0, 20.0]), rs_min_pct_short=rng.choice([5.0, 10.0]),
        rs_top_n_long=rng.choice([0, 0, 2, 5]), rs_top_n_short=rng.choice([0, 0, 2, 3]),
        enable_shorts=rng.random() > 0.5,
        near_high_max_pct=rng.choice([0.0, 0.05, 0.10, 0.20]), recent_rs_filter=rng.random() > 0.3,
        recent_rs_days=rng.choice([5, 10]), recent_rs_min_pct=rng.choice([0.0, 2.0]),
        ema21_rising_filter=rng.random() > 0.3, ema21_slope_min_pct=rng.choice([0.5, 1.0]),
        lower_high_filter=rng.random() > 0.3, lower_high_prior_days=rng.choice([5, 10]),
    )


def test_watchlist_matches_original_stage_for_stage():
    mod = oracle.load_original()
    reasons = collections.Counter()
    sizes, sides = [], collections.Counter()
    for seed in range(60):
        rng = random.Random(seed)
        cfg = random_cfg(rng)
        candles, vols = universe(rng, bench=rng.random() > 0.15)       # sometimes no benchmark data at all
        want = original_watchlist(mod, cfg, candles, vols)
        got, dropped = my_watchlist(cfg, candles, vols)
        assert got == want, f"seed {seed}\n  original: {want}\n  port:     {got}\n  cfg: {cfg}"
        sizes.append(len(want))
        sides.update(side for _s, side in want)
        for r in dropped.values():
            reasons[r.split(" ")[0] + " " + r.split(" ")[1] if " " in r else r] += 1
    # the scenarios were not trivial: non-empty watchlists, both sides, and every stage dropped something
    assert sum(1 for s in sizes if s) >= 30 and max(sizes) >= 3
    assert sides["LONG"] > 0 and sides["SHORT"] > 0 and sides[None] > 0
    for needle in ("traded", "ADR", "no", "RS", "daily", "cut", "making"):
        assert any(k.startswith(needle) for k in reasons), f"no symbol was ever dropped by '{needle}' ({dict(reasons)})"


def _stock_with_30d_return(ret, n=250, p=100.0):
    """Flat history, then a straight line to `ret` over the last 30 completed days; in-progress candle = last close."""
    closes = [p] * (n - 31) + [p * (1 + ret * k / 30.0) for k in range(31)] + [p * (1 + ret)]
    cs = []
    for i, c in enumerate(closes[:n]):
        cs.append({"t": i * 86_400_000, "o": c, "h": c * 1.03, "l": c * 0.97, "c": c, "v": 1})
    return cs


def test_leader_laggard_overlap_resolves_to_leader_like_the_original():
    """With rs_min_pct_long = -15 and rs_min_pct_short = 10 a stock at -12 pts is BOTH a leader and a laggard;
    the original calls it a leader (laggards = laggards - leaders)."""
    mod = oracle.load_original()
    cfg = dataclasses.replace(StrategyConfig(), min_turnover_inr=FLOOR, adr_min_pct=0.0, rs_min_pct_long=-15.0,
                              rs_min_pct_short=10.0, enable_shorts=True, near_high_max_pct=0.0,
                              recent_rs_filter=False, ema21_rising_filter=False, lower_high_filter=False)
    candles = {"xyz:SP500": _stock_with_30d_return(0.0), "xyz:A": _stock_with_30d_return(-0.12),
               "xyz:B": _stock_with_30d_return(-0.30), "xyz:C": _stock_with_30d_return(0.05)}
    vols = {"A": FLOOR * 2, "B": FLOOR * 2, "C": FLOOR * 2}
    # the daily-ribbon stage would drop flat/declining shapes, so compare the RS stage itself
    cfg = dataclasses.replace(cfg, ema_fast=1, ema_mid=2, ema_slow=3)
    want = original_watchlist(mod, cfg, candles, vols)
    got, _ = my_watchlist(cfg, candles, vols)
    assert got == want
    sides = dict(want)
    assert sides.get("A") == "LONG"                      # the overlapping stock resolves to LONG, not SHORT
