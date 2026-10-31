"""Portfolio simulation of the EMA-ribbon pullback bot on NSE stocks.

Time model (same idea as the original author's backtester)
    The live bot polls every 30 minutes.  We step over BASE bars (daily, or hourly when you have
    intraday data) and at the OPEN of every bar run one bot cycle with live price = open[i]:

        stop-fill detection -> exit check (once per newly completed signal candle)
        -> TP2 upkeep -> scale-in limit upkeep -> entry limit upkeep

    Then the bar's high/low is replayed against the resting orders, ADVERSE FIRST:

        stops -> entry-limit fill -> add-limit fill -> take-profits
        (stops are re-checked after every fill)

    Only data known at the open of bar i is used: completed candles + open[i] as the live price.

What is shared across symbols (the original backtester simulated symbols independently)
    one account: equity = capital + realised P&L (net of costs) + unrealised;
    an entry/add needs free margin = equity - sum(position value) / leverage.

Known limits (also listed in the README): a daily base gives ONE decision point per day instead of
the live bot's ~13 polls; if one bar both fills an entry and touches its stop the stop is taken;
no circuit-limit / T2T / ASM-GSM modelling; no dividends (use adjusted data); costs are flat bps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import rules
from .entry import evaluate_entry, select_setup
from .config import ExecConfig, StrategyConfig
from .data import resample_session_bars
from .frames import DAY_MS, HOUR_MS, Frame
from .selection import Selection, regime_allowed, select_symbol

LONG, SHORT = rules.LONG, rules.SHORT


# ───────────────────────────── records ─────────────────────────────

@dataclass
class Fill:
    symbol: str
    ts: int
    action: str            # ENTRY / ADD / TP1 / TP2 / STOP / EXIT
    direction: str
    price: float
    units: int
    fee: float
    pnl: float = 0.0       # gross P&L realised by this fill
    note: str = ""


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    avg_entry: float
    exit_price: float      # units-weighted average exit
    size_inr: float        # initial notional
    max_units: int
    adds: int
    reason: str            # reason of the final closing fill
    partials: str          # e.g. "TP1,TP2"
    gross_pnl: float
    fees: float

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees


class _Pos:
    def __init__(self, side, units, px, i, ts, init_inr, band):
        self.side, self.units, self.avg, self.i, self.ts = side, units, px, i, ts
        self.first_px, self.init_inr, self.band = px, init_inr, band
        self.sign = 1 if side == LONG else -1
        self.adds = 0
        self.tp1_done = self.tp2_done = False
        self.stops: List[dict] = []     # {"price", "units" (None = whole position)}
        self.tp1: Optional[dict] = None
        self.tp2: Optional[dict] = None
        self.gross = self.fees = 0.0
        self.exit_units = 0
        self.exit_notional = 0.0
        self.partials: List[str] = []
        self.max_units = units


def _slip(px: float, is_buy: bool, bps: float) -> float:
    return px * (1 + bps / 1e4) if is_buy else px * (1 - bps / 1e4)


def _tick(price: float, tick: float) -> float:
    """Round to the NSE tick (Rs 0.05)."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


# ───────────────────────────── per-symbol state ─────────────────────────────

class _Sym:
    def __init__(self, symbol: str, base: Frame, D: Frame, S: Frame, P: Frame):
        self.symbol = symbol
        self.F, self.D, self.S, self.P = base, D, S, P
        self.n = len(base)
        self.idx_by_ts = {t: i for i, t in enumerate(base.t_open)}
        self.pos: Optional[_Pos] = None
        self.pending_entry: Optional[dict] = None
        self.pending_add: Optional[dict] = None
        self.last_entry_day: Optional[int] = None
        self.last_add_day: Optional[int] = None
        self.last_proc_k = -2
        self.sel: Optional[Selection] = None       # None = not on the watchlist ("volume blocked")
        self.rs_dir: Optional[str] = None          # last RS side stamped on this symbol (persists)
        self.last_rs = float("-inf")
        self.last_sel_ts: Optional[int] = None
        self.cache: Dict[tuple, object] = {}
        self.last_close = base.c[0] if base.c else 0.0
        self.watch_bars = 0


# ───────────────────────────── results ─────────────────────────────

@dataclass
class Result:
    cfg: StrategyConfig
    exe: ExecConfig
    trades: List[Trade]
    fills: List[Fill]
    ts: List[int]
    equity: List[float]
    open_positions: List[dict]
    pending_entries: List[dict]
    rejected_no_margin: int
    watch_days: Dict[str, int]
    timeline_start: int = 0


# ───────────────────────────── engine ─────────────────────────────

class Engine:
    def __init__(self, daily: Dict[str, pd.DataFrame], cfg: StrategyConfig, exe: ExecConfig,
                 bench: Optional[pd.DataFrame] = None, hourly: Optional[Dict[str, pd.DataFrame]] = None,
                 base_hours: int = 1):
        """daily : {symbol: daily OHLCV+turnover}           bench : benchmark daily OHLCV
        hourly: optional {symbol: hourly bars} -> step hourly (signal_interval may then be Nh)
        """
        self.cfg, self.exe = cfg, exe
        self.syms: Dict[str, _Sym] = {}
        self.B = Frame.from_df(bench, None) if bench is not None and len(bench) else None
        self.hourly_mode = bool(hourly)
        sig_h = None if cfg.signal_interval.lower() in ("1d", "d", "24h") else int(cfg.signal_interval.lower().rstrip("h"))
        pyr_h = None if cfg.pyramid_interval.lower() in ("1d", "d", "24h") else int(cfg.pyramid_interval.lower().rstrip("h"))
        if not self.hourly_mode and (sig_h or pyr_h):
            raise ValueError("signal/pyramid intervals shorter than 1d need hourly data (--hourly)")
        src = hourly if self.hourly_mode else daily
        for sym, df in src.items():
            if df is None or df.empty or sym not in daily or daily[sym].empty:
                continue
            D = Frame.from_df(daily[sym], None)
            if self.hourly_mode:
                F = Frame.from_df(df, base_hours)
                S = D if sig_h is None else Frame.from_df(resample_session_bars(df, sig_h), sig_h)
                P = S if pyr_h == sig_h else (D if pyr_h is None else Frame.from_df(resample_session_bars(df, pyr_h), pyr_h))
            else:
                F = D
                S = P = D
            self.syms[sym] = _Sym(sym, F, D, S, P)

        # account
        self.realized = 0.0
        self.trades: List[Trade] = []
        self.fills: List[Fill] = []
        self.rejected_no_margin = 0
        self.regime: Optional[str] = None
        self.watch_days: Dict[str, int] = {}
        self.trade_start_ms = int(pd.Timestamp(exe.trade_start).value // 1_000_000) if exe.trade_start else 0
        self.tiers = sorted([(float(p), float(f)) for p, f in cfg.stop_tiers if p > 0 and f > 0])
        if not self.tiers:
            self.tiers = [(cfg.stop_loss_pct_ema9, 1.0)]
        self.eband = rules.band_label(cfg.entry_ema)
        self.eband_s = rules.band_label(cfg.entry_ema_short, self.eband)
        self.pband = rules.band_label(cfg.pyramid_ema)
        self.pband_s = rules.band_label(cfg.pyramid_ema_short, self.pband)
        self.periods = (cfg.ema_fast, cfg.ema_mid, cfg.ema_slow)

    # ═══════════════════════ account helpers ═══════════════════════
    def equity(self) -> float:
        unreal = sum(s.pos.sign * s.pos.units * (s.last_close - s.pos.avg) for s in self.syms.values() if s.pos)
        return self.exe.capital + self.realized + unreal

    def free_margin(self) -> float:
        used = sum(s.pos.units * s.last_close for s in self.syms.values() if s.pos) / max(self.cfg.leverage, 1e-9)
        return self.equity() - used

    def _fee(self, inr: float) -> float:
        return inr * self.exe.fee_bps_per_side / 1e4

    def _log(self, s: _Sym, ts, action, px, units, fee, pnl=0.0, note=""):
        self.fills.append(Fill(s.symbol, ts, action, s.pos.side, px, units, fee, pnl, note))

    # ═══════════════════════ position lifecycle ═══════════════════════
    def _open(self, s: _Sym, i: int, side: str, px: float, inr: float, band: str) -> bool:
        units = int(math.floor(inr / px))
        if units < 1:
            return False
        if units * px / self.cfg.leverage > self.free_margin():
            self.rejected_no_margin += 1
            return False
        fee = self._fee(units * px)
        ts = s.F.t_open[i]
        s.pos = _Pos(side, units, px, i, ts, units * px, band)
        s.pos.fees += fee
        self.realized -= fee
        self._log(s, ts, "ENTRY", px, units, fee)
        return True

    def _reduce(self, s: _Sym, i: int, units: int, raw_px: float, action: str, taker: bool, note=""):
        p = s.pos
        units = min(units, p.units)
        if units <= 0:
            return
        is_buy = p.side == SHORT
        px = _slip(raw_px, is_buy, self.exe.slippage_bps) if taker else raw_px
        fee = self._fee(units * px)
        gross = p.sign * units * (px - p.avg)
        p.units -= units
        p.gross += gross
        p.fees += fee
        p.exit_units += units
        p.exit_notional += units * px
        self.realized += gross - fee
        self._log(s, s.F.t_open[i], action, px, units, fee, gross, note)
        if p.units <= 0:
            self._finish(s, i, note or action)
        else:
            p.partials.append(action)

    def _finish(self, s: _Sym, i: int, reason: str):
        p = s.pos
        self.trades.append(Trade(
            s.symbol, p.side, p.ts, s.F.t_open[i], p.first_px, p.avg,
            p.exit_notional / p.exit_units if p.exit_units else p.avg,
            p.init_inr, p.max_units, p.adds, reason, ",".join(p.partials), p.gross, p.fees))
        s.pos = None
        s.pending_add = None

    def _market_close(self, s: _Sym, i: int, raw_px: float, reason: str):
        self._reduce(s, i, s.pos.units, raw_px, "EXIT", True, reason)

    # ═══════════════════════ stops / take-profits ═══════════════════════
    def _stop_pct(self, band: str) -> float:
        c = self.cfg
        return {"EMA9": c.stop_loss_pct_ema9, "EMA21": c.stop_loss_pct_ema21,
                "EMA50": c.stop_loss_pct_ema50}.get(band, c.stop_loss_pct_ema9)

    def _set_tiered_stops(self, s: _Sym):
        p = s.pos
        p.stops = []
        if len(self.tiers) > 1:
            for pct, frac in self.tiers:
                u = int(math.floor(p.units * frac))
                if u > 0:
                    p.stops.append({"price": _tick(p.avg * (1 - p.sign * pct), self.cfg.tick_size), "units": u})
        else:
            pct = self.tiers[-1][0]
            p.stops.append({"price": _tick(p.avg * (1 - p.sign * pct), self.cfg.tick_size), "units": None})

    def _place_tp1(self, s: _Sym, from_i: int):
        p, c = s.pos, self.cfg
        if c.tp_profit_pct <= 0 or c.tp_size_pct <= 0 or p.tp1_done:
            return
        units = int(math.floor(p.units * c.tp_size_pct))
        if units < 1:
            p.tp1 = None
            return
        p.tp1 = {"price": _tick(p.avg * (1 + p.sign * c.tp_profit_pct), c.tick_size), "units": units, "from_i": from_i}

    # ═══════════════════════ per-day cached daily inputs ═══════════════════════
    def _day(self, s: _Sym, i: int) -> int:
        return s.F.t_open[i] // DAY_MS

    def _daily_ribbon(self, s: _Sym, i: int):
        key = ("dr", self._day(s, i))
        if key not in s.cache:
            ts, x = s.F.t_open[i], s.F.o[i]
            kd = s.D.done_index(ts)
            s.cache[key] = s.D.triple_live(kd, x, self.periods) if kd >= 0 else (None,) * 3
        return s.cache[key]

    def _extremes(self, s: _Sym, i: int):
        """(near-extreme stats or None, lower-high stats or None) from COMPLETED daily closes."""
        key = ("ne", self._day(s, i))
        if key not in s.cache:
            kd = s.D.done_index(s.F.t_open[i])
            c = self.cfg
            need = max(c.near_high_lookback, c.lower_high_recent_days + c.lower_high_prior_days) + 1
            done = s.D.closes_done(kd, need)
            s.cache[key] = (rules.near_extreme_stats(done, c.near_high_lookback),
                            rules.lower_high_stats(done, c.lower_high_recent_days, c.lower_high_prior_days))
        return s.cache[key]

    def _tp2_inputs(self, s: _Sym, i: int):
        key = ("tp2", self._day(s, i))
        if key not in s.cache:
            kd = s.D.done_index(s.F.t_open[i])
            e50 = s.D.ema_done(self.cfg.ema_slow, kd)
            lo = max(0, kd + 1 - self.cfg.ema50_tp_adr_period)
            s.cache[key] = (e50, rules.adr_pct(s.D.h[lo:kd + 1], s.D.l[lo:kd + 1]))
        return s.cache[key]

    # ═══════════════════════ selection ═══════════════════════
    def _update_selection(self, s: _Sym, i: int):
        ts = s.F.t_open[i]
        if s.last_sel_ts is not None and ts - s.last_sel_ts < self.cfg.universe_refresh_hours * HOUR_MS:
            return
        s.last_sel_ts = ts
        kd = s.D.done_index(ts)
        kb = self.B.done_index(ts) if self.B else -1
        r = select_symbol(s.D, kd, s.F.o[i], self.B, kb, self.cfg)
        if r.ok:
            s.sel = r
            s.rs_dir = r.direction
            s.last_rs = r.rs if r.rs is not None else 0.0
        else:
            s.sel = None
            s.last_rs = float("-inf")

    # ═══════════════════════ setups ═══════════════════════
    def _emas(self, s: _Sym, F: Frame, i: int):
        """(live triple, done triple, last completed close) of frame F at the open of base bar i."""
        ts, x = s.F.t_open[i], s.F.o[i]
        k = F.done_index(ts)
        if k < 0:
            return None
        return F.triple_live(k, x, self.periods), F.triple_done(k, self.periods), F.c[k], k

    def _select_setup(self, s: _Sym, live, i: int, band_for):
        direction, band, price, _why = select_setup(
            self.cfg, live, band_for, self.regime, s.rs_dir,
            self._daily_ribbon(s, i) if self.cfg.daily_ribbon_confirm else (None, None, None))
        return direction, band, price

    # ═══════════════════════ cycle pieces ═══════════════════════
    def _check_exit(self, s: _Sym, i: int, ks: int, done, c_close) -> bool:
        p, c = s.pos, self.cfg
        closes_done = s.S.closes_done(ks, c.window_candles - 1)
        reason = rules.check_exit(p.side == LONG, closes_done, *done, c.exit_mode, c.exit_profit_2closes,
                                  c.exit_lower_close_max_closes, *self.periods)
        if reason:
            self._market_close(s, i, s.F.o[i], reason)
            return True
        return False

    def _maintain_tp2(self, s: _Sym, i: int):
        p, c = s.pos, self.cfg
        if c.ema50_tp_mult <= 0 or c.ema50_tp_size_pct <= 0 or not p.tp1_done or p.tp2_done:
            return
        day = self._day(s, i)
        if p.tp2 and p.tp2["day"] == day:
            return
        e50, adr = self._tp2_inputs(s, i)
        if e50 is None or adr is None:
            return
        target = e50 * (1 + p.sign * c.ema50_tp_mult * adr / 100.0)
        if target <= 0:
            return
        target = _tick(target, c.tick_size)
        units = int(math.floor(p.units * c.ema50_tp_size_pct))
        if units < 1:
            return
        if p.tp2 and abs(target / p.tp2["price"] - 1) < 0.01 and abs(units - p.tp2["units"]) <= p.tp2["units"] * 0.02:
            p.tp2["day"] = day
            return
        live = s.F.o[i]
        if (live >= target) if p.sign > 0 else (live <= target):   # marketable reduce-only limit -> fills now
            p.tp2 = None
            p.tp2_done = True
            self._reduce(s, i, units, live, "TP2", True, "TP2")
            return
        p.tp2 = {"price": target, "units": units, "day": day, "from_i": i}

    def _maintain_add(self, s: _Sym, i: int):
        p, c = s.pos, self.cfg
        today = self._day(s, i)
        if p.init_inr <= 0:
            return
        if s.last_add_day == today or p.adds >= c.max_additional_adds:
            s.pending_add = None
            return
        em = self._emas(s, s.P, i)
        if em is None:
            s.pending_add = None
            return
        live, done, prev_close, _ = em
        add_band = self.pband_s if p.side == SHORT else self.pband
        direction, band, limit = self._select_setup(s, live, i, lambda d: add_band)
        if band is None or direction != p.side:
            s.pending_add = None
            return
        if not rules.closed_candle_confirms(direction, prev_close, *done, add_band):
            s.pending_add = None
            return
        x = s.F.o[i]
        prof = (x - p.avg) / p.avg if p.side == LONG else (p.avg - x) / p.avg
        if prof < c.pyramid_min_profit_pct:
            s.pending_add = None
            return
        if rules.would_fill_immediately(direction, limit, x):
            s.pending_add = None
            return
        pa = s.pending_add
        stale = pa and (s.F.t_open[i] - pa["t"]) >= c.limit_refresh_hours * HOUR_MS
        if pa and pa["band"] == band and not stale:
            return
        pct = rules.add_size_pct(c.additional_entry_size_pct, c.additional_entry_size_step_pct,
                                 c.additional_entry_size_min_pct, p.adds)
        s.pending_add = {"price": _tick(limit, c.tick_size), "band": band, "inr": p.init_inr * pct,
                         "t": s.F.t_open[i], "from_i": i}

    def _maintain_entry(self, s: _Sym, i: int):
        c = self.cfg
        today = self._day(s, i)
        ts = s.F.t_open[i]
        if s.last_entry_day == today or s.sel is None or ts < self.trade_start_ms:
            s.pending_entry = None
            return
        em = self._emas(s, s.S, i)
        if em is None or em[3] < c.ema_slow:            # live bot: needs EMA_SLOW + 2 candles
            s.pending_entry = None
            return
        live, done, prev_close, _ = em
        near, lh = self._extremes(s, i)
        direction, band, limit, _why = evaluate_entry(
            c, live=live, done=done, prev_close=prev_close, daily_ribbon=self._daily_ribbon(s, i),
            near=near, lower_high=lh, x=s.F.o[i], rs_dir=s.rs_dir, regime=self.regime,
            eband=self.eband, eband_s=self.eband_s)
        if direction is None:
            s.pending_entry = None
            return
        pe = s.pending_entry
        stale = pe and (ts - pe["t"]) >= c.limit_refresh_hours * HOUR_MS
        if pe and pe["side"] == direction and pe["band"] == band and not stale:
            return
        inr = c.initial_entry_inr_short if direction == SHORT else c.initial_entry_inr_long
        s.pending_entry = {"side": direction, "band": band, "price": _tick(limit, c.tick_size), "inr": inr,
                           "t": ts, "from_i": i}

    # ═══════════════════════ intrabar replay ═══════════════════════
    def _touch(self, s: _Sym, i: int, is_long_order: bool, price: float) -> bool:
        return s.F.l[i] <= price if is_long_order else s.F.h[i] >= price

    def _intrabar(self, s: _Sym, i: int):
        for _ in range(12):                           # re-run after any fill that changes orders
            if not self._intrabar_pass(s, i):
                break

    def _intrabar_pass(self, s: _Sym, i: int) -> bool:
        F, o, today = s.F, s.F.o[i], self._day(s, i)
        p = s.pos
        # 1. stops
        if p and p.stops:
            hit = [st for st in p.stops if (F.l[i] <= st["price"] if p.sign > 0 else F.h[i] >= st["price"])]
            if hit:
                hit.sort(key=lambda st: -st["price"] * p.sign)       # nearest to the market first
                for st in hit:
                    if not s.pos:
                        break
                    gap = o <= st["price"] if p.sign > 0 else o >= st["price"]
                    px = o if gap else st["price"]                    # gap through the stop fills at the open
                    units = s.pos.units if st["units"] is None else min(st["units"], s.pos.units)
                    s.pos.stops.remove(st)
                    self._reduce(s, i, units, px, "STOP", True, "STOP")
                    s.last_entry_day = today                          # no same-day re-entry after a stop
                    if s.pos is None:
                        s.pending_add = None
                return True
        # 2. entry limit
        pe = s.pending_entry
        if not p and pe and pe["from_i"] <= i and self._touch(s, i, pe["side"] == LONG, pe["price"]):
            s.pending_entry = None
            if self._open(s, i, pe["side"], pe["price"], pe["inr"], pe["band"]):
                s.last_entry_day = s.last_add_day = today
                self._set_tiered_stops(s)
                self._place_tp1(s, i + 1)
            return True
        # 3. scale-in limit
        pa = s.pending_add
        if p and pa and pa["from_i"] <= i and self._touch(s, i, p.side == LONG, pa["price"]):
            self._fill_add(s, i, pa)
            return True
        # 4. take-profits
        if p:
            for key, action in (("tp1", "TP1"), ("tp2", "TP2")):
                tp = getattr(p, key)
                if tp and tp["from_i"] <= i and (F.h[i] >= tp["price"] if p.sign > 0 else F.l[i] <= tp["price"]):
                    setattr(p, key, None)
                    setattr(p, key + "_done", True)
                    marketable = o >= tp["price"] if p.sign > 0 else o <= tp["price"]
                    px, taker = (o, True) if marketable else (tp["price"], False)
                    self._reduce(s, i, tp["units"], px, action, taker, action)
                    if s.pos:
                        if key == "tp1":
                            self._after_tp1(s)
                    else:
                        s.last_entry_day = today
                    return True
        return False

    def _after_tp1(self, s: _Sym):
        p = s.pos
        if self.cfg.tp_stop_to_breakeven:
            p.stops = [{"price": _tick(p.avg, self.cfg.tick_size), "units": None}]
        else:   # keep the original stop price, resized to the runner
            p.stops = [{"price": st["price"], "units": None} for st in p.stops[-1:]] or p.stops
        if p.tp2:
            p.tp2["day"] = None

    def _fill_add(self, s: _Sym, i: int, pa: dict):
        p, c = s.pos, self.cfg
        px = pa["price"]
        units = int(math.floor(pa["inr"] / px))
        s.pending_add = None
        if units < 1:
            return
        if units * px / c.leverage > self.free_margin():
            self.rejected_no_margin += 1
            return
        fee = self._fee(units * px)
        p.avg = (p.units * p.avg + units * px) / (p.units + units)
        p.units += units
        p.max_units = max(p.max_units, p.units)
        p.fees += fee
        p.adds += 1
        self.realized -= fee
        s.last_add_day = self._day(s, i)
        self._log(s, s.F.t_open[i], "ADD", px, units, fee)
        # tiers end; ONE stop re-based on the new average (or at the average = breakeven)
        stop = p.avg if c.stop_to_new_avg_after_add else p.avg * (1 - p.sign * self._stop_pct(p.band))
        p.stops = [{"price": _tick(stop, c.tick_size), "units": None}]
        p.tp1 = None
        self._place_tp1(s, i + 1)
        if p.tp2:
            p.tp2["day"] = None

    # ═══════════════════════ main loop ═══════════════════════
    def run(self) -> Result:
        c = self.cfg
        timeline = sorted({t for s in self.syms.values() for t in s.F.t_open})
        ts_out: List[int] = []
        eq_out: List[float] = []
        for ts in timeline:
            if self.B is not None and c.regime_filter_enabled:
                self.regime = regime_allowed(self.B, self.B.done_index(ts), c, self.regime)
            active = [(s, s.idx_by_ts[ts]) for s in self.syms.values() if ts in s.idx_by_ts]
            active.sort(key=lambda a: -a[0].last_rs)              # strongest RS first, as the live loop
            for s, i in active:
                self._update_selection(s, i)
                if s.sel is not None:
                    s.watch_bars += 1
                em = self._emas(s, s.S, i)
                exited = False
                if em is not None:
                    live, done, c_close, ks = em
                    new_candle = ks != s.last_proc_k
                    if s.pos and new_candle:
                        exited = self._check_exit(s, i, ks, done, c_close)
                    s.last_proc_k = ks
                if not exited:
                    if s.pos:
                        self._maintain_tp2(s, i)
                        if s.pos:
                            self._maintain_add(s, i)
                    else:
                        self._maintain_entry(s, i)
                self._intrabar(s, i)
                s.last_close = s.F.c[i]
            ts_out.append(ts)
            eq_out.append(self.equity())

        opens = []
        for s in self.syms.values():
            if s.pos:
                p = s.pos
                opens.append({"symbol": s.symbol, "side": p.side, "units": p.units, "avg": p.avg,
                              "mark": s.last_close, "unrealized": p.sign * p.units * (s.last_close - p.avg),
                              "stops": [dict(x) for x in p.stops], "tp1": p.tp1, "tp2": p.tp2, "adds": p.adds,
                              "entry_ts": p.ts})
        pend = [{"symbol": s.symbol, **s.pending_entry} for s in self.syms.values() if s.pending_entry and not s.pos]
        watch_days = {s.symbol: s.watch_bars for s in self.syms.values() if s.watch_bars}
        return Result(self.cfg, self.exe, self.trades, self.fills, ts_out, eq_out, opens, pend,
                      self.rejected_no_margin, watch_days, timeline[0] if timeline else 0)
