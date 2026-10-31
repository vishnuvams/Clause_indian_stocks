"""Differential tests: the ORIGINAL `EMAPullbackTrader` vs `indian_stocks.trader.Trader`.

Both are driven through the same fake exchange with the same candle feed and the same intraday price path
(4 polls per day).  After EVERY poll the resting orders (side, size, price, reduce-only, trigger), the position,
and the trader's own state (adds, TP flags, stop price, pending orders, tier stops, same-day gates) must be
identical.  Each family also asserts that it really exercised the behaviour it targets.
"""
import random

import pytest

from tests import oracle
from tests import differential as D

pytestmark = pytest.mark.skipif(not oracle.original_available(), reason="original bot not found (set EMA_BOT_PATH)")

UP = ((0.007, 0.022), (-0.004, 0.02))
DOWN = ((-0.007, 0.022), (0.004, 0.02))
CHOP = ((0.0, 0.03),)


@pytest.fixture(scope="module")
def mod():
    return oracle.load_original()


def scenario(mod, seed, *, over=None, rs="LONG", regimes=UP, switch=80, gap=0.0, partial=1.0, hook=None,
             pos=None, setup=None, n=170, hist_n=130, polls=4, paused=False):
    rng = random.Random(seed)
    bars = D.gen_bars(rng, hist_n + n, regimes=regimes, switch_every=switch, gap_prob=gap)
    hist, future = bars[:hist_n], bars[hist_n:]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):                    # the original is chatty
        pair = D.Pair(mod, hist, over=over, rs_direction=rs, initial_position=pos, partial_frac=partial, setup=setup)
        pair.hook, pair.drawdown_paused = hook, paused
        msg = pair.run(future, rng, polls)
    return pair, msg


def family(mod, seeds, need=(), **kw):
    total = D.collections.Counter()
    for seed in seeds:
        pair, msg = scenario(mod, seed, **kw)
        assert msg is None, f"seed {seed}: {msg}"
        total.update(pair.events)
    for ev in need:
        assert total[ev] > 0, f"the family never exercised '{ev}' ({dict(total)})"
    return total


def test_default_stocks_config(mod):
    family(mod, range(12), need=("position_opened", "add_filled", "tp1_filled", "tp2_filled", "exit_by_market_close"))


def test_gaps_through_the_stop(mod):
    family(mod, range(14), need=("position_opened", "closed_by_stop_or_tp"), gap=0.05)


def test_tiered_stops(mod):
    tiers = {"stop_tiers": [(0.03, 0.3333), (0.05, 0.3333), (0.07, 0.3334)]}
    family(mod, range(14), need=("position_opened", "tier_stop_fired"), over=tiers, gap=0.03,
           regimes=((0.006, 0.03), (-0.004, 0.03)))


def test_shorts_mirror(mod):
    t = family(mod, range(12), need=("position_opened",), over={"enable_shorts": True}, rs="SHORT", regimes=DOWN)
    assert t["exit_by_market_close"] + t["closed_by_stop_or_tp"] > 0


def test_both_directions_when_rs_gate_is_off(mod):
    family(mod, range(12), need=("position_opened",), over={"enable_shorts": True}, rs=None,
           regimes=((0.008, 0.022), (-0.008, 0.022)), switch=45)


def test_stop_to_new_average_after_add(mod):
    family(mod, range(10), need=("add_filled",), over={"stop_to_new_avg_after_add": True})


def test_runner_keeps_original_stop_when_breakeven_is_off(mod):
    family(mod, range(10), need=("tp1_filled",), over={"tp_stop_to_breakeven": False})


@pytest.mark.parametrize("band", ["ema21", "ema50"])
def test_entry_and_pyramid_on_deeper_bands(mod, band):
    family(mod, range(10), need=("position_opened",), over={"entry_ema": band, "pyramid_ema": band})


@pytest.mark.parametrize("mode", ["ema9_ema50", "ema9_ema21_ema50"])
@pytest.mark.parametrize("ref", ["ema21", "ema50", "off"])
def test_exit_modes(mod, mode, ref):
    family(mod, range(6), need=("position_opened",), over={"exit_mode": mode, "exit_profit_2closes": ref})


def test_filters_switched_off(mod):
    off = {"daily_ribbon_confirm": False, "near_high_max_pct": 0.0, "limit_extension_max_pct": 0.0,
           "lower_high_filter": False}
    family(mod, range(10), need=("position_opened",), over=off, regimes=CHOP)


def test_adds_heavy_config(mod):
    family(mod, range(10), need=("add_filled",),
           over={"pyramid_min_profit_pct": 0.03, "max_additional_adds": 3, "tp_profit_pct": 0.5})


def test_take_profits_disabled(mod):
    t = family(mod, range(8), need=("position_opened",), over={"tp_profit_pct": 0.0, "ema50_tp_mult": 0.0})
    assert t["tp1_filled"] == 0


def test_long_refresh_interval_keeps_resting_limits(mod):
    family(mod, range(8), need=("position_opened",), over={"limit_refresh_hours": 24})


def test_regime_gate_toggles(mod):
    def hook(day, pair):
        pair.set_all(regime_allowed=[ "LONG", None, "SHORT", "LONG"][(day // 15) % 4])
    family(mod, range(10), need=("position_opened",), hook=hook, over={"enable_shorts": True}, rs=None,
           regimes=((0.008, 0.022), (-0.008, 0.022)), switch=40)


def test_watchlist_membership_and_drawdown_pause(mod):
    def hook(day, pair):
        pair.set_all(_volume_blocked=(day // 12) % 3 == 1)
        pair.drawdown_paused = (day // 17) % 4 == 2
    family(mod, range(10), need=("position_opened",), hook=hook)


def test_partially_filled_entries_and_adds(mod):
    family(mod, range(12), need=("position_opened",), partial=0.4)


# ───────────────────────── restart-sync: adopt a position the tracker does not know ─────────────────────────

def _stop_orders(prices_and_sizes, is_buy=False):
    def setup(w):
        for px, sz in prices_and_sizes:
            w.hl.order(w.hl.coin, is_buy, sz, px * 0.95, {"trigger": {"triggerPx": px, "isMarket": True, "tpsl": "sl"}},
                       reduce_only=True)
    return setup


@pytest.mark.parametrize("stops,label", [
    ([], "no stop"),
    ([(0.9, 1)], "one valid stop"),
    ([(1.3, 1)], "one stop on the wrong side"),
    ([(0.9, 1), (0.85, 1)], "duplicate stops"),
])
def test_restart_sync(mod, stops, label):
    for seed in range(4):
        rng = random.Random(seed)
        bars = D.gen_bars(rng, 130 + 100)
        entry = bars[129][3] * 0.9
        px = [(entry * f, 500.0) for f, _ in stops]
        pair, msg = scenario(mod, seed, pos=(500.0, entry), setup=_stop_orders(px), n=100)
        assert msg is None, f"{label}, seed {seed}: {msg}"


def test_restart_sync_of_a_winner_does_not_replace_tp1(mod):
    rng = random.Random(1)
    bars = D.gen_bars(rng, 230)
    entry = bars[129][3] / 1.3                                   # already +30%: TP1 must not be re-placed
    pair, msg = scenario(mod, 1, pos=(500.0, entry), setup=_stop_orders([(entry * 0.9, 500.0)]), n=100)
    assert msg is None, msg
    assert pair.worlds[0].trader.partial_tp_done or pair.events["tp1_filled"] == 0


# ───────────────────────── intraday: 1h signal ribbon / 4h-1h pyramid timeframe, 30-minute polls ─────────────────────────

def intraday(mod, seed, *, over=None, rs="LONG", regimes=UP, days=26, hist_days=14, gap=0.0):
    import contextlib, io
    rng = random.Random(seed)
    # the daily-ribbon / near-high gates need 50+ daily candles of history, so use a long daily history
    bars = D.gen_bars(rng, 130 + days, regimes=regimes, switch_every=60, gap_prob=gap)
    with contextlib.redirect_stdout(io.StringIO()):
        pair = D.IntradayPair(mod, bars[:130], rng, over=over, rs_direction=rs)
        msg = pair.run(bars[130:], rng)
    return pair, msg


def intraday_family(mod, seeds, need=(), **kw):
    total = D.collections.Counter()
    for seed in seeds:
        pair, msg = intraday(mod, seed, **kw)
        assert msg is None, f"seed {seed}: {msg}"
        total.update(pair.events)
    for ev in need:
        assert total[ev] > 0, f"never exercised '{ev}' ({dict(total)})"


def test_1h_signal_with_daily_pyramid_timeframe(mod):
    intraday_family(mod, range(4), need=("position_opened",), over={"signal_interval": "1h"},
                    regimes=((0.012, 0.03), (-0.006, 0.03)))


def test_1h_signal_with_1h_pyramid_timeframe(mod):
    intraday_family(mod, range(4), need=("position_opened",),
                    over={"signal_interval": "1h", "pyramid_interval": "1h", "pyramid_min_profit_pct": 0.04},
                    regimes=((0.012, 0.03), (-0.006, 0.03)))


def test_daily_signal_with_4h_pyramid_timeframe(mod):
    intraday_family(mod, range(4), need=("position_opened",),
                    over={"signal_interval": "1d", "pyramid_interval": "4h", "pyramid_min_profit_pct": 0.04},
                    regimes=((0.012, 0.03), (-0.006, 0.03)))


SLOW_EXIT = {"exit_mode": "ema9_ema21_ema50", "exit_profit_2closes": "off"}      # let hourly winners run
STRONG = ((0.02, 0.025), (-0.01, 0.03))


def test_1h_signal_adds_take_profits_and_stops(mod):
    over = {"signal_interval": "1h", "pyramid_min_profit_pct": 0.03, "tp_profit_pct": 0.06, "ema50_tp_mult": 1.0,
            **SLOW_EXIT}
    intraday_family(mod, range(5), need=("position_opened", "add_filled", "tp1_filled"), over=over,
                    regimes=STRONG, days=45, gap=0.04)


def test_1h_pyramid_timeframe_adds_and_second_take_profit(mod):
    over = {"signal_interval": "1h", "pyramid_interval": "1h", "pyramid_min_profit_pct": 0.03,
            "tp_profit_pct": 0.05, "ema50_tp_mult": 0.5, **SLOW_EXIT}
    intraday_family(mod, range(5), need=("position_opened", "add_filled", "tp1_filled", "tp2_filled"), over=over,
                    regimes=STRONG, days=45)


def test_4h_pyramid_timeframe_adds(mod):
    over = {"signal_interval": "1d", "pyramid_interval": "4h", "pyramid_min_profit_pct": 0.03, "tp_profit_pct": 0.3,
            **SLOW_EXIT}
    intraday_family(mod, range(5), need=("position_opened", "add_filled"), over=over, regimes=STRONG, days=45)


def test_1h_shorts_mirror(mod):
    over = {"signal_interval": "1h", "enable_shorts": True, "pyramid_min_profit_pct": 0.03, "tp_profit_pct": 0.06,
            **SLOW_EXIT}
    intraday_family(mod, range(4), need=("position_opened",), over=over, rs="SHORT",
                    regimes=((-0.02, 0.025), (0.01, 0.03)), days=40)
