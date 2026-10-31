import numpy as np

from indian_stocks import rules
from indian_stocks.config import StrategyConfig
from indian_stocks.frames import Frame
from indian_stocks.selection import regime_allowed, select_symbol
from tests.helpers import flat, make_df, trend

CFG = StrategyConfig()


# ───────────────────────── rules ─────────────────────────

def test_add_size_sequence():
    seq = [rules.add_size_pct(0.5, 0.10, 0.10, n) for n in range(8)]
    assert [round(x, 2) for x in seq] == [0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]


def test_weak_run_examples():
    # LONG, EMA3: price falls below the EMA, the 2nd closing is lower than the 1st -> lower-close exit
    closes = [10, 10, 10, 10, 10, 9, 8.5]
    trig, run, kind, now, prev = rules.weak_run_decision(closes, 3, True, 3)
    assert (trig, kind, now, prev) == (True, "lower_close", 8.5, 9)
    # a close back above the EMA ends the run: no exit
    closes = [10, 10, 10, 10, 9, 11]
    assert rules.weak_run_decision(closes, 3, True, 3)[0] is False
    # single close below the EMA is not an exit yet (needs a lower close or the time limit)
    closes = [10, 10, 10, 10, 10, 9.9]
    assert rules.weak_run_decision(closes, 3, True, 3)[0] is False
    # SHORT mirror
    assert rules.weak_run_decision([10, 10, 10, 10, 10, 11, 11.5], 3, False, 3)[2] == "lower_close"


def test_weak_run_time_limit():
    # three closes below the EMA, each HIGHER than the previous (recovering) -> only the time limit fires
    closes = [10] * 10 + [8.0, 8.1, 8.2]
    trig, run, kind, *_ = rules.weak_run_decision(closes, 3, True, 3)
    assert (trig, kind, run) == (True, "time_limit", 3)
    assert rules.weak_run_decision(closes[:-1], 3, True, 3)[0] is False


def test_ribbon_and_daily_gates():
    assert rules.ribbon_ok(3, 2, 1) and rules.ribbon_ok(1, 2, 3)
    assert not rules.ribbon_ok(2, 3, 1) and not rules.ribbon_ok(None, 2, 1)
    assert rules.ribbon_ok(2, 1, None) and not rules.ribbon_ok(1, 1, None)       # young stock: 2-EMA test
    assert rules.daily_ribbon_confirms("LONG", 3, 2, 1) and not rules.daily_ribbon_confirms("LONG", 3, 3, 1)
    assert rules.daily_ribbon_confirms("SHORT", 1, 2, 3)
    assert rules.daily_ribbon_confirms("LONG", 3, 2, None) and not rules.daily_ribbon_confirms("LONG", 3, None, 1)


def test_raw_setup_requires_full_ribbon_and_respects_shorts():
    band = lambda d: "EMA21" if d == "SHORT" else "EMA9"
    assert rules.raw_setup(3, 2, 1, False, band) == ("LONG", "EMA9", 3)
    assert rules.raw_setup(1, 2, 3, False, band) == (None, None, None)           # long-only
    assert rules.raw_setup(1, 2, 3, True, band) == ("SHORT", "EMA21", 2)
    assert rules.raw_setup(3, 2, None, True, band) == (None, None, None)


def test_do_not_chase():
    assert rules.would_fill_immediately("LONG", 100, 99) and rules.would_fill_immediately("LONG", 100, 100)
    assert not rules.would_fill_immediately("LONG", 100, 101)
    assert rules.would_fill_immediately("SHORT", 100, 101) and not rules.would_fill_immediately("SHORT", 100, 99)


def test_closed_candle_confirms_band_logic():
    # band EMA21 only needs a close above EMA21 and EMA50 (price may sit under EMA9 on a pullback)
    assert rules.closed_candle_confirms("LONG", 105, 110, 100, 90, "EMA21")
    assert not rules.closed_candle_confirms("LONG", 105, 110, 100, 90, "EMA9")
    assert not rules.closed_candle_confirms("LONG", 95, 110, 100, 90, "EMA21")


# ───────────────────────── selection ─────────────────────────

def _frames(n=200, **kw):
    return Frame.from_df(trend(n, **kw), None), Frame.from_df(flat(n), None)


def _sel(D, B, cfg=CFG, k=None):
    k = len(D) - 1 if k is None else k
    return select_symbol(D, k, D.c[k], B, k, cfg)


def test_strong_leader_is_selected_long():
    D, B = _frames()
    r = _sel(D, B)
    assert r.ok and r.direction == "LONG" and r.rs > 0.10 and r.adr > 4.0


def test_volume_floor():
    D, B = _frames(volume=100)                      # ~Rs 10-70k traded value a day
    r = _sel(D, B)
    assert not r.ok and "traded value" in r.reason


def test_adr_floor():
    D, B = _frames(spread=0.001)                    # ADR ~0.2%
    r = _sel(D, B)
    assert not r.ok and r.reason.startswith("ADR")


def test_rs_floor_benchmark_matters():
    D = Frame.from_df(trend(200, g=0.004), None)    # +12% in 30d
    flat_b = Frame.from_df(flat(200), None)
    assert _sel(D, flat_b).ok
    rising_b = Frame.from_df(trend(200, g=0.004, spread=0.004), None)           # benchmark rises as fast
    r = _sel(D, rising_b)
    assert not r.ok and r.reason.startswith("RS")


def test_laggard_not_selected_when_shorts_off_and_selected_when_on():
    n = 200
    closes = [100 * (0.99 ** i) for i in range(n)]
    opens = [100.0] + closes[:-1]
    df = make_df(opens, [max(o, c) * 1.025 for o, c in zip(opens, closes)],
                 [min(o, c) * 0.975 for o, c in zip(opens, closes)], closes, volume=100_000_000)
    D, B = Frame.from_df(df, None), Frame.from_df(flat(n), None)
    off = _sel(D, B)
    assert not off.ok and off.reason.startswith("RS")          # a laggard is simply not a long candidate
    import dataclasses
    r = _sel(D, B, dataclasses.replace(CFG, enable_shorts=True, near_high_max_pct=0.0))
    assert r.ok and r.direction == "SHORT"


def test_ribbon_filter_drops_tangled_daily_ema():
    # strong 30d return but the last days roll over -> EMA9 < EMA21 even though RS is high
    n = 200
    closes = [100 * 1.01 ** i for i in range(150)] + [100 * 1.01 ** 149 * 0.985 ** j for j in range(1, 51)]
    opens = [100.0] + closes[:-1]
    df = make_df(opens, [max(o, c) * 1.025 for o, c in zip(opens, closes)],
                 [min(o, c) * 0.975 for o, c in zip(opens, closes)], closes)
    D, B = Frame.from_df(df, None), Frame.from_df(flat(n), None)
    k = 160                                         # a few days into the roll-over
    r = select_symbol(D, k, D.c[k], B, k, CFG)
    assert not r.ok


def test_lower_high_and_near_high_filters():
    n = 120
    base = [100 * 1.01 ** i for i in range(n)]
    # make days 1-5 ago lower than days 6-10 ago
    closes = base[:]
    for j in range(1, 6):
        closes[-j] = base[-6] * 0.97
    opens = [100.0] + closes[:-1]
    df = make_df(opens, [max(o, c) * 1.025 for o, c in zip(opens, closes)],
                 [min(o, c) * 0.975 for o, c in zip(opens, closes)], closes)
    D, B = Frame.from_df(df, None), Frame.from_df(flat(n), None)
    import dataclasses
    off = dataclasses.replace(CFG, near_high_max_pct=0.0, recent_rs_filter=False, ema21_rising_filter=False,
                              lower_high_filter=False)
    assert select_symbol(D, n - 1, D.c[n - 1], B, n - 1, off).ok
    r = select_symbol(D, n - 1, D.c[n - 1], B, n - 1, dataclasses.replace(off, lower_high_filter=True))
    assert not r.ok and "lower highs" in r.reason
    r = select_symbol(D, n - 1, D.c[n - 1], B, n - 1, dataclasses.replace(off, near_high_max_pct=0.02))
    assert not r.ok and "closing high" in r.reason


def test_regime_follows_benchmark_vs_ema21():
    import dataclasses
    cfg = dataclasses.replace(CFG, regime_filter_enabled=True)
    up, dn = Frame.from_df(trend(100), None), Frame.from_df(make_df(
        [100 * 0.99 ** i for i in range(100)], [101 * 0.99 ** i for i in range(100)],
        [99 * 0.99 ** i for i in range(100)], [100 * 0.99 ** (i + 1) for i in range(100)]), None)
    assert regime_allowed(up, 99, cfg, None) == "LONG"
    assert regime_allowed(dn, 99, cfg, None) == "SHORT"
    assert regime_allowed(up, 99, CFG, "SHORT") is None                   # filter off
    assert regime_allowed(None, 99, cfg, "LONG") is None
