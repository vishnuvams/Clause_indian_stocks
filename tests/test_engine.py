"""Engine scenarios on hand-built tapes.  Every expected number is derived independently in the test."""
import dataclasses
import math

import pandas as pd
import pytest

from indian_stocks.config import ExecConfig, StrategyConfig
from indian_stocks.engine import Engine
from indian_stocks.frames import Frame
from tests.helpers import flat, make_df, random_walk, trend

CFG = StrategyConfig()
EXE = ExecConfig(capital=5_000_000, fee_bps_per_side=12.0, slippage_bps=5.0, trade_start="2021-03-01")
T = 100                       # the bar on which the pullback entry is engineered
N = 190


def tick(p, t=0.05):
    return round(round(p / t) * t, 10)


class Tape:
    """Mutable OHLC lists on top of a steady uptrend + helpers to read the engine's own EMAs."""

    def __init__(self, n=N, g=0.01):
        df = trend(n, g=g)
        self.o, self.h, self.l, self.c = (df[k].tolist() for k in ("open", "high", "low", "close"))
        self.n = n

    def df(self):
        return make_df(self.o, self.h, self.l, self.c)

    def frame(self):
        return Frame.from_df(self.df(), None)

    def limit_at(self, j):
        """The EMA9 limit the bot would rest at the open of bar j (live EMA incl. open[j])."""
        F = self.frame()
        return tick(F.ema_live(9, j - 1, self.o[j]))

    def pullback(self, j):
        lim = self.limit_at(j)
        self.l[j] = min(self.l[j], lim * 0.99)
        return lim

    def set_bar(self, j, o, c, up=1.005, dn=0.995):
        self.o[j], self.c[j] = o, c
        self.h[j] = max(o, c) * up
        self.l[j] = min(o, c) * dn


def run(tape, cfg=CFG, exe=EXE, bench=None):
    eng = Engine({"AAA": tape.df()}, cfg, exe, bench if bench is not None else flat(tape.n))
    return eng, eng.run()


def fills_of(res, action):
    return [f for f in res.fills if f.action == action]


def when(res_ts):
    return pd.Timestamp(res_ts * 1_000_000)


# ───────────────────────────── entry ─────────────────────────────

def test_plain_uptrend_never_pulls_back_to_ema9_so_no_entry():
    _, res = run(Tape())
    assert res.fills == [] and res.trades == []
    assert res.watch_days["AAA"] > 50                       # it WAS on the watchlist, just never filled


def test_pullback_fills_at_the_ema9_limit_with_minus10pct_stop_and_tp1():
    t = Tape()
    lim = t.pullback(T)
    eng, res = run(t)
    entries = fills_of(res, "ENTRY")
    assert len(entries) == 1
    e = entries[0]
    assert e.direction == "LONG" and e.price == lim and when(e.ts).date() == t.df().index[T].date()
    units = math.floor(CFG.initial_entry_inr_long / lim)
    assert e.units == units
    assert e.fee == pytest.approx(units * lim * 12 / 1e4)
    # TP1: 30% of the position at +20% (resting limit => maker fill at its own price)
    tp1 = fills_of(res, "TP1")
    assert len(tp1) == 1
    assert tp1[0].price == tick(lim * 1.20) and tp1[0].units == math.floor(units * 0.30)
    # after TP1 the runner's stop is at breakeven (the average entry)
    pos = res.open_positions[0]
    assert pos["stops"] == [{"price": tick(lim), "units": None}]


def test_initial_stop_is_10pct_below_fill_and_gap_through_it_fills_at_the_open():
    t = Tape()
    lim = t.pullback(T)
    stop = tick(lim * 0.90)
    gap_open = lim * 0.85
    t.set_bar(T + 1, gap_open, gap_open)
    for j in range(T + 2, t.n):                             # then sit flat so nothing else happens
        t.set_bar(j, gap_open, gap_open)
    eng, res = run(t)
    stp = fills_of(res, "STOP")
    assert len(stp) == 1 and stp[0].ts == eng.syms["AAA"].F.t_open[T + 1]
    assert stp[0].price == pytest.approx(gap_open * (1 - 5 / 1e4))        # gapped through: open, minus taker slippage
    assert gap_open < stop
    tr = res.trades[0]
    units = math.floor(CFG.initial_entry_inr_long / lim)
    gross = units * (stp[0].price - lim)
    assert tr.gross_pnl == pytest.approx(gross)
    assert tr.fees == pytest.approx(units * lim * 12e-4 + units * stp[0].price * 12e-4)
    assert tr.reason == "STOP"


def test_stop_inside_the_bar_fills_at_the_stop_price():
    t = Tape()
    lim = t.pullback(T)
    stop = tick(lim * 0.90)
    t.o[T + 1] = lim * 1.02
    t.c[T + 1] = lim * 0.95
    t.h[T + 1] = lim * 1.03
    t.l[T + 1] = stop * 0.99                                 # trades through the stop intrabar
    for j in range(T + 2, t.n):
        t.set_bar(j, lim * 0.95, lim * 0.95)
    _, res = run(t)
    stp = fills_of(res, "STOP")[0]
    assert stp.price == pytest.approx(stop * (1 - 5 / 1e4))


def test_no_entry_when_price_is_already_past_the_limit():
    """If the open is at/below the EMA9 limit it would fill at market -> the bot does not place it."""
    t = Tape()
    F = t.frame()
    j = T
    t.o[j] = F.ema_done(9, j - 1) * 0.95                     # open far below the EMA9
    t.l[j] = t.o[j] * 0.99
    t.c[j] = t.o[j]
    live_ema9 = tick(Frame.from_df(t.df(), None).ema_live(9, j - 1, t.o[j]))
    assert t.o[j] <= live_ema9                               # precondition: the limit WOULD fill instantly
    assert t.l[j] <= live_ema9                               # ...and without the guard the bar would fill it
    eng, res = run(t)
    assert [f for f in fills_of(res, "ENTRY") if f.ts == eng.syms["AAA"].F.t_open[j]] == []


# ───────────────────────────── exits ─────────────────────────────

def test_lower_close_exit_runs_at_the_next_open():
    t = Tape()
    lim = t.pullback(T)
    # first close under EMA9, then a LOWER close under EMA9 -> exit at the open after the second one
    e9 = t.frame().ema_done(9, T)
    c1 = e9 * 0.985
    t.set_bar(T + 1, t.c[T], c1)
    c2 = c1 * 0.99
    t.set_bar(T + 2, c1, c2)
    nxt = c2 * 0.995
    t.set_bar(T + 3, nxt, nxt)
    for j in range(T + 4, t.n):
        t.set_bar(j, nxt, nxt)
    eng, res = run(t)
    ex = fills_of(res, "EXIT")
    assert len(ex) == 1
    assert ex[0].ts == eng.syms["AAA"].F.t_open[T + 3]
    assert ex[0].price == pytest.approx(nxt * (1 - 5 / 1e4))
    assert res.trades[0].reason.startswith("lower close while below EMA9")


def test_a_single_close_under_ema9_does_not_exit():
    t = Tape()
    t.pullback(T)
    e9 = t.frame().ema_done(9, T)
    t.set_bar(T + 1, t.c[T], e9 * 0.99)
    for j in range(T + 2, t.n):                              # next close back above EMA9 and rising
        t.set_bar(j, t.c[j - 1], t.c[j - 1] * 1.01)
    _, res = run(t)
    assert fills_of(res, "EXIT") == []


# ───────────────────────────── scale-in ─────────────────────────────

def test_scale_in_needs_20pct_profit_then_adds_50pct_and_rebases_the_stop():
    t = Tape(n=170)
    lim = t.pullback(T)
    # first bar where the open is >= 25% above the entry => profit gate (20%) is satisfied
    j = next(k for k in range(T + 1, t.n) if t.o[k] >= lim * 1.25)
    add_lim = t.pullback(j)
    eng, res = run(t)
    # an earlier pullback bar (profit < 20%) must NOT have produced an add
    adds = fills_of(res, "ADD")
    assert len(adds) == 1 and adds[0].ts == eng.syms["AAA"].F.t_open[j]
    entry_units = math.floor(CFG.initial_entry_inr_long / lim)
    init_value = entry_units * lim
    assert adds[0].price == add_lim
    assert adds[0].units == math.floor(init_value * 0.50 / add_lim)          # first add = 50% of the initial
    pos = res.open_positions[0]
    assert pos["adds"] == 1
    # tiers/breakeven are replaced by ONE stop 3% under the NEW average
    assert pos["stops"] == [{"price": tick(pos["avg"] * 0.97), "units": None}]


def test_no_add_below_20pct_profit():
    t = Tape(n=140)
    lim = t.pullback(T)
    j = next(k for k in range(T + 1, t.n) if t.o[k] >= lim * 1.08)           # only ~+8%
    t.pullback(j)
    t.n = j + 2
    df = t.df().iloc[:j + 2]
    res = Engine({"AAA": df}, CFG, EXE, flat(len(df))).run()
    assert len(fills_of(res, "ENTRY")) == 1                  # the position existed and the pullback touched EMA9
    assert fills_of(res, "ADD") == []


# ───────────────────────────── account ─────────────────────────────

def test_entry_rejected_when_free_margin_is_too_small():
    t = Tape()
    t.pullback(T)
    _, res = run(t, exe=dataclasses.replace(EXE, capital=50_000))
    assert res.rejected_no_margin >= 1 and res.trades == [] and res.fills == []


def test_leverage_lets_the_same_capital_take_the_trade():
    t = Tape()
    t.pullback(T)
    cfg = dataclasses.replace(CFG, leverage=5.0)
    _, res = run(t, cfg=cfg, exe=dataclasses.replace(EXE, capital=250_000))
    assert len(fills_of(res, "ENTRY")) == 1


def test_one_entry_per_symbol_per_day_and_none_after_a_stop_the_same_day():
    t = Tape()
    lim = t.pullback(T)
    t.l[T] = lim * 0.80                                       # dips through the -10% stop on the entry bar itself
    eng, res = run(t)
    stop_day = [f for f in res.fills if f.action == "STOP"]
    assert len(stop_day) == 1 and stop_day[0].ts == eng.syms["AAA"].F.t_open[T]   # conservative: stop is taken
    per_day = {}
    for f in fills_of(res, "ENTRY"):
        per_day.setdefault(pd.Timestamp(f.ts * 10**6).date(), 0)
        per_day[pd.Timestamp(f.ts * 10**6).date()] += 1
    assert all(v == 1 for v in per_day.values())


# ───────────────────────────── invariants on random tapes ─────────────────────────────

RELAXED = dataclasses.replace(
    CFG, rs_enabled=False, adr_min_pct=0.0, min_turnover_inr=0.0, near_high_max_pct=0.0, recent_rs_filter=False,
    ema21_rising_filter=False, lower_high_filter=False, limit_extension_max_pct=0.0, enable_shorts=True,
    initial_entry_inr_long=300_000, initial_entry_inr_short=150_000)


def _random_book(seeds=(1, 2, 3, 4, 5, 6), n=500):
    return {f"S{s}": random_walk(n, s, drift=0.0008 if s % 2 else -0.0003, vol=0.025) for s in seeds}


def test_accounting_identity_and_trade_fill_consistency_on_random_book():
    daily = _random_book()
    exe = dataclasses.replace(EXE, capital=3_000_000, trade_start="2021-06-01")
    eng = Engine(daily, RELAXED, exe, flat(500))
    res = eng.run()
    assert len(res.trades) >= 10, "relaxed filters should produce plenty of trades to check"
    assert {t.direction for t in res.trades} == {"LONG", "SHORT"}

    # equity = capital + closed net P&L + open positions' (gross - fees + unrealised)
    open_part = sum(s.pos.gross - s.pos.fees + s.pos.sign * s.pos.units * (s.last_close - s.pos.avg)
                    for s in eng.syms.values() if s.pos)
    closed = sum(t.net_pnl for t in res.trades)
    assert res.equity[-1] == pytest.approx(exe.capital + closed + open_part, rel=1e-9)

    # every closed trade: the fills inside its [entry, exit] window add up to the Trade record
    for tr in res.trades:
        fl = [f for f in res.fills if f.symbol == tr.symbol and tr.entry_ts <= f.ts <= tr.exit_ts]
        assert tr.gross_pnl == pytest.approx(sum(f.pnl for f in fl), rel=1e-9, abs=1e-6)
        assert tr.fees == pytest.approx(sum(f.fee for f in fl), rel=1e-9, abs=1e-6)
        assert tr.exit_ts >= tr.entry_ts and tr.max_units >= 1


def test_no_overlapping_positions_no_negative_units_and_one_entry_per_day():
    eng = Engine(_random_book(), RELAXED, dataclasses.replace(EXE, capital=3_000_000), flat(500))
    res = eng.run()
    by_sym = {}
    for t in res.trades:
        by_sym.setdefault(t.symbol, []).append(t)
    for trs in by_sym.values():
        trs.sort(key=lambda t: t.entry_ts)
        for a, b in zip(trs, trs[1:]):
            assert a.exit_ts <= b.entry_ts                       # never two positions in one symbol
    assert all(f.units >= 1 for f in res.fills)
    seen = set()
    for f in fills_of(res, "ENTRY"):
        key = (f.symbol, pd.Timestamp(f.ts * 10**6).date())
        assert key not in seen
        seen.add(key)


def test_shorts_are_off_by_default():
    cfg = dataclasses.replace(RELAXED, enable_shorts=False)
    res = Engine(_random_book(), cfg, dataclasses.replace(EXE, capital=3_000_000), flat(500)).run()
    assert res.trades and {t.direction for t in res.trades} == {"LONG"}


def test_trade_start_blocks_earlier_entries():
    exe = dataclasses.replace(EXE, capital=3_000_000, trade_start="2022-06-01")
    res = Engine(_random_book(), RELAXED, exe, flat(500)).run()
    assert res.trades
    assert min(t.entry_ts for t in res.trades) >= int(pd.Timestamp("2022-06-01").value // 1_000_000)
