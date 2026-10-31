"""`Trader` - the per-stock decision engine (LONG-ONLY), a broker-agnostic port of `EMAPullbackTrader`.

Same control flow, same order of operations, same state, same caches.  What it does NOT contain: the
Hyperliquid SDK, Telegram, trade-log files, printing, rate-limit back-off, sleeps.  All venue I/O goes through
the `Broker` protocol below; price/size rounding through `Venue`.  `tests/test_oracle_*.py` drive the ORIGINAL
class and this one on the same fake exchange and require identical resting orders, positions and stops after
every cycle.

One `check_strategy()` call == one poll of the original bot (it polls every 30 min):

    1. candles of the signal timeframe (250, the last one still forming) -> "live" EMAs (incl. forming candle)
       and "closed" EMAs (completed candles only)
    2. stop-fill detection: single stop, then the tiered stops
    3. pending entry-limit fill  ->  _on_entry_limit_filled
    4. pending add-limit fill    ->  _on_additional_filled
    5. restart-sync (adopt a position the tracker does not know about)
    6. defensive stop reconciliation, then "at most one stop / limit / TP" enforcement
    7. status / flat-reason (kept: it warms the once-per-day caches exactly as the original does)
    8. EXIT check (once per newly completed candle)
    9. take-profit fills, TP2 upkeep
   10. scale-in limit upkeep (while open)  /  entry limit upkeep (while flat)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from . import rules
from .config import StrategyConfig

LONG = rules.LONG


# ───────────────────────────── interfaces ─────────────────────────────

@dataclass
class OrderView:
    oid: int
    is_buy: bool
    size: float
    price: float
    reduce_only: bool = False
    is_trigger: bool = False
    trigger_px: float = 0.0


class Broker(Protocol):
    """Everything the trader needs from a venue (one symbol, one account)."""

    def candles(self, interval: str, count: int) -> List[dict]:
        """OHLCV dicts with t,o,h,l,c (numbers); oldest first; the LAST one is still forming."""

    def position(self) -> dict:
        """{"size": signed units (0 if flat), "entry": average entry price, "pnl": unrealised P&L}"""

    def orders(self) -> List[OrderView]:
        """All resting orders on this symbol (limits and stop triggers)."""

    def place_limit(self, is_buy: bool, size: float, price: float, reduce_only: bool = False):
        """BUY entries / scale-ins and reduce-only SELL take-profits -> ("resting", oid) | ("filled", None) | None"""

    def place_stop(self, size: float, trigger_px: float, limit_px: float):
        """reduce-only SELL stop: triggers at `trigger_px`, sells down to `limit_px`
        -> ("resting", oid) | ("filled", None) | None"""

    def cancel(self, oid: int) -> None: ...

    def market_close(self, size: float, ref_price: float) -> bool:
        """Sell the whole position now (limit at `ref_price`, which sits below the market)."""


@dataclass
class Venue:
    """Venue price/size rules.  Defaults = NSE equities: Rs 0.05 tick, whole shares (rounded down)."""
    tick: float = 0.05
    fmt_price: Optional[Callable[[float], float]] = None
    round_size: Optional[Callable[[float], float]] = None

    def price(self, p: float) -> float:
        if self.fmt_price:
            return self.fmt_price(p)
        return round(round(p / self.tick) * self.tick, 10) if self.tick > 0 else p

    def size(self, s: float) -> float:
        if self.round_size:
            return self.round_size(s)
        return float(math.floor(abs(s)))


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ist_today() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")


# ───────────────────────────── the trader ─────────────────────────────

class Trader:
    def __init__(self, broker: Broker, cfg: StrategyConfig, venue: Optional[Venue] = None, *,
                 now: Callable[[], float], today: Callable[[], str] = ist_today,
                 log: Optional[Callable[[str], None]] = None,
                 initial_entry_long: Optional[float] = None):
        self.b = broker
        self.cfg = cfg
        self.venue = venue or Venue(tick=cfg.tick_size)
        self._now, self._today = now, today
        self.say = log or (lambda m: None)

        # strategy params (see EMAPullbackTrader.__init__)
        self.EMA_FAST, self.EMA_MID, self.EMA_SLOW = cfg.ema_fast, cfg.ema_mid, cfg.ema_slow
        self.STOP_PCT_BY_BAND = {"EMA9": cfg.stop_loss_pct_ema9, "EMA21": cfg.stop_loss_pct_ema21,
                                 "EMA50": cfg.stop_loss_pct_ema50}
        self.STOP_TIERS = sorted([(float(p), float(f)) for p, f in cfg.stop_tiers if float(p) > 0 and float(f) > 0],
                                 key=lambda t: t[0]) or [(cfg.stop_loss_pct_ema9, 1.0)]
        self.LIMIT_REFRESH_SEC = cfg.limit_refresh_hours * 3600
        self.ADDITIONAL_ENTRY_SIZE_PCT = cfg.additional_entry_size_pct
        self.ADD_SIZE_STEP_PCT = max(0.0, float(cfg.additional_entry_size_step_pct))
        self.ADD_SIZE_MIN_PCT = max(0.0, float(cfg.additional_entry_size_min_pct))
        self.MAX_ADDITIONAL_ADDS = cfg.max_additional_adds
        self.PYRAMID_MIN_PROFIT_PCT = cfg.pyramid_min_profit_pct
        _pi = str(cfg.pyramid_interval).strip().lower()
        self.pyramid_interval = _pi if _pi in ("1d", "4h", "1h") else "1d"
        self.signal_interval = cfg.signal_interval
        self.TP_PROFIT_PCT, self.TP_SIZE_PCT, self.TP_STOP_TO_BE = cfg.tp_profit_pct, cfg.tp_size_pct, cfg.tp_stop_to_breakeven
        self.entry_band_label = rules.band_label(cfg.entry_ema)
        self.pyramid_band_label = rules.band_label(cfg.pyramid_ema)
        self.stop_to_new_avg_after_add = bool(cfg.stop_to_new_avg_after_add)
        self.daily_ribbon_confirm = bool(cfg.daily_ribbon_confirm)
        self.near_high_max_pct = max(0.0, float(cfg.near_high_max_pct))
        self.near_high_lookback = max(2, int(cfg.near_high_lookback))
        self.limit_extension_max_pct = max(0.0, float(cfg.limit_extension_max_pct))
        self.lower_high_filter = bool(cfg.lower_high_filter)
        self.lh_recent = max(1, int(cfg.lower_high_recent_days))
        self.lh_prior = max(1, int(cfg.lower_high_prior_days))
        self.initial_entry_long = cfg.initial_entry_inr_long if initial_entry_long is None else initial_entry_long
        self.exit_mode = str(cfg.exit_mode).strip().lower()
        _e2c = str(cfg.exit_profit_2closes).strip().lower()
        self.exit_profit_2closes = _e2c if _e2c in ("ema50", "ema21", "ema9") else "off"
        self.exit_lower_close_max_closes = max(2, int(cfg.exit_lower_close_max_closes))
        self.EMA50_TP_MULT = max(0.0, float(cfg.ema50_tp_mult))
        self.EMA50_TP_SIZE_PCT = min(1.0, max(0.0, float(cfg.ema50_tp_size_pct)))
        self.EMA50_TP_ADR_PERIOD = max(3, int(cfg.ema50_tp_adr_period))

        # once-per-day caches
        self._daily_ribbon_cache = None
        self._daily_ribbon_cache_date = None
        self._near_extreme_cache = None
        self._near_extreme_cache_date = None
        self._lower_high_cache = None
        self._tp2_cache = None
        self._tp2_cache_date = None

        # gates set by the orchestrator
        self.regime_blocks_longs = False        # benchmark below its EMA21 (only when the regime filter is on)
        self._volume_blocked = False
        self.in_scanner = True

        # position state
        self.position_type: Optional[str] = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_band: Optional[str] = None
        self.additional_adds_done = 0
        self.partial_tp_done = False
        self.initial_entry_snapshot = 0.0
        self.tp_oid = None
        self.tp_price = 0.0
        self.tp_size = 0.0
        self.tp2_oid = None
        self.tp2_price = 0.0
        self.tp2_size = 0.0
        self.tp2_done = False
        self._tp2_placed_date = None
        self._tp2_missing_cycles = 0
        self._tp1_filled_here = False
        self.last_entry_fill_date = None
        self.last_add_fill_date = None
        self.pending_entry_oid = None
        self.pending_entry_side = None
        self.pending_entry_band = None
        self.pending_entry_price = 0.0
        self.last_limit_refresh_ts = 0.0
        self._partial_stop_size = 0.0
        self.pending_add_oid = None
        self.pending_add_side = None
        self.pending_add_band = None
        self.pending_add_price = 0.0
        self.last_add_refresh_ts = 0.0
        self.emergency_stop_price: Optional[float] = None
        self.stop_loss_oid = None
        self._tiers_active = False
        self._tier_stop_oids: List[dict] = []
        self._tier_base_size = 0.0
        self.last_processed_candle_ts = 0
        self._last_ema = (None, None, None)

    # ═══════════════════════ venue helpers ═══════════════════════
    def _stop_pct_for(self, band):
        return self.STOP_PCT_BY_BAND.get(band, self.STOP_PCT_BY_BAND.get("EMA9", 0.05))

    def get_candles(self, interval="1d", count=250):
        c = self.b.candles(interval, count)
        return c[-count:] if c else []

    def get_position(self):
        p = self.b.position()
        size = float(p.get("size", 0) or 0)
        return {"size": size, "entry": float(p.get("entry", 0) or 0) if size != 0 else 0.0,
                "pnl": float(p.get("pnl", 0) or 0)}

    def calculate_ema(self, closes, period):
        return rules.ema_value(closes, period)

    # ═══════════════════════ orders ═══════════════════════
    def get_trigger_orders(self) -> List[OrderView]:
        return [o for o in self.b.orders() if o.is_trigger]

    def cancel_all_trigger_orders(self):
        for o in self.get_trigger_orders():
            if o.oid:
                self.b.cancel(o.oid)

    def place_stop_loss_order(self, size, trigger_price):
        if trigger_price <= 0:
            return None
        size = self.venue.size(abs(size))
        trigger_price = self.venue.price(trigger_price)
        limit_price = self.venue.price(trigger_price * 0.95)
        res = self.b.place_stop(size, trigger_price, limit_price)
        if res is None:
            return None
        kind, oid = res
        return oid if kind == "resting" else "filled"

    def _ensure_stop_loss_placed(self, size, trigger_price, max_attempts=3, track=True):
        for _ in range(max_attempts):
            oid = self.place_stop_loss_order(size, trigger_price)
            valid = isinstance(oid, int) or (isinstance(oid, str) and oid not in ("placed", "filled"))
            if valid:
                if track:
                    self.stop_loss_oid = oid
                return oid
            tos = self.get_trigger_orders()
            if tos:
                known = {t["oid"] for t in self._tier_stop_oids}
                if not track and self.stop_loss_oid:
                    known.add(self.stop_loss_oid)
                cand = next((o.oid for o in tos if o.oid not in known), None)
                if cand:
                    if track:
                        self.stop_loss_oid = cand
                    return cand
        self.say(f"STOP PLACEMENT FAILED - position unprotected @ {trigger_price}")
        return None

    def _replace_stop(self, size, trigger_price):
        self.cancel_all_trigger_orders()
        self.stop_loss_oid = None
        return self._ensure_stop_loss_placed(size, trigger_price)

    def cancel_stop_loss_order(self):
        if not self.stop_loss_oid or self.stop_loss_oid in ("placed", "filled"):
            for o in self.get_trigger_orders():
                if o.oid:
                    self.b.cancel(o.oid)
            return
        self.b.cancel(self.stop_loss_oid)
        self.stop_loss_oid = None

    def _get_non_trigger_limits(self, include_reduce_only=False) -> List[OrderView]:
        out = []
        for o in self.b.orders():
            if o.is_trigger:
                continue
            if not include_reduce_only and o.reduce_only:
                continue
            out.append(o)
        return out

    def _clear_pending_entry(self):
        self.pending_entry_oid = None
        self.pending_entry_side = None
        self.pending_entry_band = None
        self.pending_entry_price = 0.0

    def _clear_pending_add(self):
        self.pending_add_oid = None
        self.pending_add_side = None
        self.pending_add_band = None
        self.pending_add_price = 0.0

    def _sweep_non_trigger_limits(self, reason=""):
        orders = self._get_non_trigger_limits()
        cancelled = 0
        for o in orders:
            if not o.oid:
                continue
            self.b.cancel(o.oid)
            cancelled += 1
        if cancelled > 0:
            self._clear_pending_entry()
            self._clear_pending_add()
        return cancelled

    # ---- entry limit ----
    def _place_entry_limit(self, limit_price, usd_amount, completed_ts, band):
        self._sweep_non_trigger_limits("pre-entry-limit sweep")
        size = self.venue.size(usd_amount / limit_price)
        limit_price = self.venue.price(limit_price)
        res = self.b.place_limit(True, size, limit_price)
        if res is None:
            return None
        kind, oid = res
        self.pending_entry_oid = oid if kind == "resting" else "filled"
        self.pending_entry_side, self.pending_entry_band, self.pending_entry_price = LONG, band, limit_price
        return self.pending_entry_oid

    def _check_entry_limit_filled(self):
        if not self.pending_entry_oid:
            return False, None
        if self.pending_entry_oid == "filled":
            return True, self.pending_entry_price
        still = any(str(o.oid) == str(self.pending_entry_oid) for o in self.b.orders())
        if still:
            return False, None
        pos = self.get_position()
        if pos["size"] != 0:
            return True, pos["entry"]
        self._clear_pending_entry()
        return False, None

    def _cancel_entry_limit(self):
        if not self.pending_entry_oid:
            return
        if self.pending_entry_oid in ("placed", "filled"):
            self._clear_pending_entry()
            return
        self.b.cancel(self.pending_entry_oid)
        self._clear_pending_entry()

    # ---- add limit ----
    def _place_additional_limit(self, limit_price, usd_amount, band):
        self._sweep_non_trigger_limits("pre-additional-limit sweep")
        size = self.venue.size(usd_amount / limit_price)
        limit_price = self.venue.price(limit_price)
        res = self.b.place_limit(True, size, limit_price)
        if res is None:
            return None
        kind, oid = res
        self.pending_add_oid = oid if kind == "resting" else "filled"
        self.pending_add_side, self.pending_add_band, self.pending_add_price = LONG, band, limit_price
        return self.pending_add_oid

    def _check_additional_limit_filled(self):
        if not self.pending_add_oid:
            return False, None
        if self.pending_add_oid == "filled":
            return True, self.pending_add_price
        still = any(str(o.oid) == str(self.pending_add_oid) for o in self.b.orders())
        if still:
            return False, None
        return True, self.pending_add_price

    def _cancel_additional_limit(self):
        if not self.pending_add_oid:
            return
        if self.pending_add_oid in ("placed", "filled"):
            self._clear_pending_add()
            return
        self.b.cancel(self.pending_add_oid)
        self._clear_pending_add()

    # ---- take profits ----
    def _place_tp_limit(self, size, tp_price):
        size = self.venue.size(abs(size))
        tp_price = self.venue.price(tp_price)
        if size <= 0 or tp_price <= 0:
            return None
        res = self.b.place_limit(False, size, tp_price, reduce_only=True)
        if res is None:
            return None
        kind, oid = res
        self.tp_oid = oid if kind == "resting" else "filled"
        self.tp_price, self.tp_size = tp_price, size
        return self.tp_oid

    def _place_tp_for_position(self):
        if self.TP_PROFIT_PCT <= 0 or self.TP_SIZE_PCT <= 0:
            return
        if self.partial_tp_done or not self.position_type or self.entry_price <= 0:
            return
        tp_price = self.entry_price * (1 + self.TP_PROFIT_PCT)
        tp_size = self.venue.size(self.position_size * self.TP_SIZE_PCT)
        if self.tp_oid:
            self._cancel_tp_limit()
        self._place_tp_limit(tp_size, tp_price)

    def _cancel_tp_limit(self):
        oid = self.tp_oid
        self.tp_oid = None
        self.tp_price = 0.0
        self.tp_size = 0.0
        if oid and oid not in ("placed", "filled"):
            self.b.cancel(oid)
        self._cancel_tp2_limit()

    def _cancel_tp2_limit(self):
        oid = self.tp2_oid
        self.tp2_oid = None
        self.tp2_price = 0.0
        self.tp2_size = 0.0
        self._tp2_placed_date = None
        if not oid or oid in ("placed", "filled"):
            return
        self.b.cancel(oid)

    def _tp2_inputs(self):
        today = self._today()
        if self._tp2_cache is None or self._tp2_cache_date != today:
            candles = self.get_candles("1d", 250)
            if not candles or len(candles) < 3:
                return None
            done = candles[:-1]
            closes = [float(c["c"]) for c in done]
            rows = done[-self.EMA50_TP_ADR_PERIOD:]
            self._tp2_cache = (self.calculate_ema(closes, self.EMA_SLOW),
                               rules.adr_pct([float(c["h"]) for c in rows], [float(c["l"]) for c in rows]))
            self._tp2_cache_date = today
        return self._tp2_cache

    def _tp2_target(self):
        data = self._tp2_inputs()
        if not data or data[0] is None or data[1] is None:
            return None
        ema50, adr = data
        target = ema50 * (1.0 + self.EMA50_TP_MULT * adr / 100.0)
        return target if target > 0 else None

    def _place_tp2_limit(self, size, price):
        size = self.venue.size(abs(size))
        price = self.venue.price(price)
        if size <= 0 or price <= 0:
            return None
        res = self.b.place_limit(False, size, price, reduce_only=True)
        if res is None:
            return None
        kind, oid = res
        self.tp2_oid = oid if kind == "resting" else "filled"
        self.tp2_price, self.tp2_size = price, size
        self._tp2_placed_date = self._today()
        return self.tp2_oid

    def _maintain_tp2(self, live_price):
        if (self.EMA50_TP_MULT <= 0 or self.EMA50_TP_SIZE_PCT <= 0 or not self.position_type
                or not self.partial_tp_done or self.tp2_done):
            return
        if self.tp2_oid == "filled":
            return
        today = self._today()
        if self.tp2_oid and self._tp2_placed_date == today:
            return
        target = self._tp2_target()
        if target is None:
            return
        beyond = bool(live_price) and live_price >= target
        if beyond and not self._tp1_filled_here:
            return
        current = abs(self.get_position().get("size") or 0)
        size = self.venue.size(current * self.EMA50_TP_SIZE_PCT)
        if size <= 0:
            return
        if self.tp2_oid and self.tp2_oid != "filled":
            same_price = self.tp2_price > 0 and abs(target / self.tp2_price - 1.0) < 0.01
            if same_price and abs(size - self.tp2_size) <= max(self.tp2_size * 0.02, 0):
                self._tp2_placed_date = today
                return
            self._cancel_tp2_limit()
        self._place_tp2_limit(size, target)

    def _resize_stop_to_remaining(self, remaining):
        if self._tiers_active or remaining <= 0 or not self.emergency_stop_price or not self.position_type:
            return
        self._replace_stop(remaining, self.emergency_stop_price)

    def check_tp2_filled(self):
        if not self.position_type or not self.tp2_oid or self.tp2_done:
            return False
        if self.tp2_oid == "filled":
            gone = True
        else:
            gone = not any(str(o.oid) == str(self.tp2_oid) for o in self.b.orders())
        if not gone:
            self._tp2_missing_cycles = 0
            return False
        remaining = abs(self.get_position().get("size") or 0)
        sold = self.position_size - remaining
        if sold < self.tp2_size * 0.5:
            self._tp2_missing_cycles += 1
            if self._tp2_missing_cycles >= 2:
                self.tp2_oid = None
                self.tp2_price = 0.0
                self.tp2_size = 0.0
                self._tp2_placed_date = None
                self._tp2_missing_cycles = 0
            return False
        self.tp2_done = True
        self.tp2_oid = None
        self.tp2_price = 0.0
        self.tp2_size = 0.0
        self._tp2_placed_date = None
        self._tp2_missing_cycles = 0
        if remaining <= 0:
            try:
                self._cancel_additional_limit()
            except Exception:
                pass
            self._reset_position_state()
            self.last_entry_fill_date = self._today()
            return True
        self.position_size = remaining
        self._resize_stop_to_remaining(remaining)
        return True

    def check_tp_filled(self):
        if not self.position_type or not self.tp_oid or self.partial_tp_done:
            return False
        if self.tp_oid == "filled":
            filled = True
        else:
            filled = not any(str(o.oid) == str(self.tp_oid) for o in self.b.orders())
        if not filled:
            return False
        self.partial_tp_done = True
        self._tp1_filled_here = True
        self.tp_oid = None
        self.tp_price = 0.0
        remaining_size = abs(self.get_position()["size"])
        self._tiers_active = False
        self._tier_stop_oids = []
        if remaining_size <= 0:
            try:
                self._cancel_additional_limit()
            except Exception:
                pass
            self._reset_position_state()
            self.last_entry_fill_date = self._today()
            return True
        self.position_size = remaining_size
        if self.TP_STOP_TO_BE and self.position_size > 0:
            self.emergency_stop_price = self.entry_price
            self._replace_stop(self.position_size, self.emergency_stop_price)
        else:
            self._resize_stop_to_remaining(self.position_size)
        return True

    # ---- stops ----
    def check_stop_loss_filled(self):
        if not self.position_type or self.stop_loss_oid is None:
            return False
        ours = any(o.oid == self.stop_loss_oid for o in self.get_trigger_orders())
        if ours:
            return False
        if self.get_position().get("size", 0) != 0:
            self.stop_loss_oid = None            # stop merely missing -> _defensive_stop_check re-places it
            return False
        if self.pending_add_oid:
            try:
                self._cancel_additional_limit()
            except Exception:
                pass
        try:
            self._cancel_tp_limit()
        except Exception:
            pass
        self._reset_position_state()
        self.last_entry_fill_date = self._today()
        return True

    def _check_tiered_stops_filled(self):
        if not (self._tiers_active and self.position_type and self._tier_stop_oids):
            return False
        live_oids = {o.oid for o in self.get_trigger_orders()}
        missing = [t for t in self._tier_stop_oids if t["oid"] not in live_oids]
        if not missing:
            return False
        cur_size = abs(self.get_position().get("size", 0))
        if cur_size >= self.position_size - 1e-12:
            for t in missing:
                self._tier_stop_oids.remove(t)
                trig = self.entry_price * (1 - t["pct"])
                tier_size = self.venue.size(self._tier_base_size * t["frac"])
                new_oid = self._ensure_stop_loss_placed(tier_size, trig, track=False)
                if new_oid:
                    self._tier_stop_oids.append({"oid": new_oid, "pct": t["pct"], "frac": t["frac"]})
            return False
        for t in missing:
            self._tier_stop_oids.remove(t)
        self.position_size = cur_size
        if cur_size <= 0:
            if self.pending_add_oid:
                try:
                    self._cancel_additional_limit()
                except Exception:
                    pass
            try:
                self._cancel_tp_limit()
            except Exception:
                pass
            try:
                self.cancel_all_trigger_orders()
            except Exception:
                pass
            self._reset_position_state()
            self.last_entry_fill_date = self._today()
            return True
        self.last_entry_fill_date = self._today()       # a tier stop fired -> no re-entry today
        return False

    def _defensive_stop_check(self):
        if self.position_type is None or self.position_size <= 0:
            return
        if self._tiers_active:
            return
        if self.stop_loss_oid is not None:
            return
        tos = self.get_trigger_orders()
        if tos:
            eoid, etrig = tos[0].oid, tos[0].trigger_px
            if eoid:
                self.stop_loss_oid = eoid
                if etrig:
                    self.emergency_stop_price = float(etrig)
                return
        if self.entry_price and self.emergency_stop_price:
            self._ensure_stop_loss_placed(self.position_size, self.emergency_stop_price)

    def _enforce_single_orders(self):
        def cancel_extras(group):
            if not group:
                return None
            group = sorted(group, key=lambda o: o.oid or 0)
            for o in group[:-1]:
                if o.oid:
                    self.b.cancel(o.oid)
            return group[-1].oid

        tos = self.get_trigger_orders()
        if self._tiers_active and self._tier_stop_oids:
            tracked = {t["oid"] for t in self._tier_stop_oids}
            for o in [o for o in tos if o.oid not in tracked]:
                if o.oid:
                    self.b.cancel(o.oid)
        elif len(tos) > 1:
            kept = cancel_extras(tos)
            if kept:
                self.stop_loss_oid = kept
        lims = self._get_non_trigger_limits(include_reduce_only=True)
        adds = [o for o in lims if not o.reduce_only]
        tps = [o for o in lims if o.reduce_only]
        if len(adds) > 1:
            cancel_extras(adds)
        if len(tps) > 1:
            kept_tp = cancel_extras(tps)
            if kept_tp:
                self.tp_oid = kept_tp

    # ---- close ----
    def place_close_order(self, size):
        candles = self.get_candles("1m", 1)
        if not candles:
            return False
        price = float(candles[-1]["c"]) * 0.97
        return bool(self.b.market_close(self.venue.size(abs(size)), self.venue.price(price)))

    def _market_close_position(self, reason, current_price):
        if self.position_type is None:
            return
        size_to_close = self.position_size
        ok = False
        for _ in range(3):
            if self.place_close_order(size_to_close):
                ok = True
                break
        if not ok:
            return
        self.cancel_stop_loss_order()
        if self.pending_add_oid:
            self._cancel_additional_limit()
        self._cancel_tp_limit()
        self.say(f"CLOSE {self.position_type}: {reason}")
        self._reset_position_state()

    def _reset_position_state(self):
        self.position_type = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_band = None
        self.additional_adds_done = 0
        self.partial_tp_done = False
        self.tp_oid = None
        self.tp_price = 0.0
        self.tp_size = 0.0
        self.tp2_oid = None
        self.tp2_price = 0.0
        self.tp2_size = 0.0
        self.tp2_done = False
        self._tp2_placed_date = None
        self._tp2_missing_cycles = 0
        self._tp1_filled_here = False
        self.emergency_stop_price = None
        self.stop_loss_oid = None
        self.initial_entry_snapshot = 0.0
        self._tiers_active = False
        self._tier_stop_oids = []
        self._tier_base_size = 0.0
        # last_entry_fill_date / last_add_fill_date are kept on purpose (same-day re-entry / re-add block)

    # ═══════════════════════ fills ═══════════════════════
    def _on_entry_limit_filled(self, fill_price, ema9, ema21, ema50):
        pos = self.get_position()
        init = self.initial_entry_long
        size = abs(pos["size"]) if pos["size"] != 0 else (init / fill_price)
        entry = pos["entry"] if pos["entry"] > 0 else fill_price
        band = self.pending_entry_band or self._infer_band(entry, ema9, ema21, ema50)

        self.position_type = LONG
        self.position_size = size
        self.entry_price = entry
        self._partial_stop_size = 0.0
        self.entry_band = band
        self.additional_adds_done = 0
        self.partial_tp_done = False
        self.tp2_done = False
        self.tp2_oid = None
        self._tp2_placed_date = None
        self._tp1_filled_here = False
        self.initial_entry_snapshot = init
        self.last_entry_fill_date = self._today()
        self.last_add_fill_date = self._today()          # adds begin the NEXT day

        self._tier_base_size = size
        self._tiers_active = len(self.STOP_TIERS) > 1
        deepest_pct = self.STOP_TIERS[-1][0]
        self.emergency_stop_price = entry * (1 - deepest_pct)

        self.cancel_all_trigger_orders()
        self.stop_loss_oid = None
        if self._tiers_active:
            self._tier_stop_oids = []
            for pct, frac in self.STOP_TIERS:
                tier_size = self.venue.size(size * frac)
                if tier_size <= 0:
                    continue
                trig = entry * (1 - pct)
                oid = self._ensure_stop_loss_placed(tier_size, trig, track=False)
                if oid:
                    self._tier_stop_oids.append({"oid": oid, "pct": pct, "frac": frac})
        else:
            self._ensure_stop_loss_placed(size, self.emergency_stop_price)

        self._place_tp_for_position()
        self._clear_pending_entry()

    def _on_additional_filled(self, fill_price):
        pos = self.get_position()
        new_size = abs(pos["size"]) if pos["size"] != 0 else self.position_size
        new_avg = pos["entry"] if pos["entry"] > 0 else self.entry_price
        self.position_size = new_size
        self.entry_price = new_avg
        self.additional_adds_done += 1
        self.last_add_fill_date = self._today()
        self._tiers_active = False
        self._tier_stop_oids = []
        stop_pct = self._stop_pct_for(self.entry_band)
        if self.stop_to_new_avg_after_add:
            self.emergency_stop_price = new_avg
        else:
            self.emergency_stop_price = new_avg * (1 - stop_pct)
        self._replace_stop(new_size, self.emergency_stop_price)
        self._place_tp_for_position()
        self._tp2_placed_date = None
        self._clear_pending_add()

    def _infer_band(self, price, ema9, ema21, ema50):
        if ema9 is not None and price >= ema9:
            return "EMA9"
        if ema21 is not None and price >= ema21:
            return "EMA21"
        if ema50 is not None and price >= ema50:
            return "EMA50"
        return "EMA9"

    def _restart_sync(self, ema9, ema21, ema50, completed_ts):
        if self.pending_entry_oid:
            return False
        pos = self.get_position()
        if pos["size"] <= 0 or self.position_type is not None:       # long-only: a short is never adopted
            return False
        self.position_type = LONG
        self.position_size = pos["size"]
        self.entry_price = pos["entry"]
        self._partial_stop_size = 0.0
        self.entry_band = self._infer_band(self.entry_price, ema9, ema21, ema50)
        self.additional_adds_done = 0
        notional = abs(self.entry_price * self.position_size)
        prof_frac = (pos.get("pnl", 0.0) / notional) if notional > 0 else 0.0
        self.partial_tp_done = bool(self.TP_PROFIT_PCT and self.TP_PROFIT_PCT > 0 and prof_frac >= self.TP_PROFIT_PCT)
        self.initial_entry_snapshot = self.initial_entry_long
        self.last_add_fill_date = None
        stop_pct = self.STOP_TIERS[-1][0]
        self.emergency_stop_price = self.entry_price * (1 - stop_pct)

        tos = self.get_trigger_orders()

        def stop_is_valid(trig):
            t = float(trig or 0)
            if t <= 0 or self.entry_price <= 0:
                return False
            return t < self.entry_price and abs(t - self.entry_price) / self.entry_price <= 0.20

        if len(tos) == 0:
            self._ensure_stop_loss_placed(self.position_size, self.emergency_stop_price)
        elif len(tos) == 1 and stop_is_valid(tos[0].trigger_px):
            self.stop_loss_oid = tos[0].oid
            self.emergency_stop_price = float(tos[0].trigger_px)
        else:
            self.cancel_all_trigger_orders()
            self._ensure_stop_loss_placed(self.position_size, self.emergency_stop_price)
        if self.pending_entry_oid:
            self._cancel_entry_limit()
        if not self.partial_tp_done:
            self._place_tp_for_position()
        return True

    # ═══════════════════════ gates ═══════════════════════
    def _raw_setup(self, ema9, ema21, ema50, band_label=None):
        return rules.raw_setup(ema9, ema21, ema50, band_label or self.entry_band_label)

    def _daily_ribbon_confirms(self):
        if not self.daily_ribbon_confirm:
            return True
        today = self._today()
        if self._daily_ribbon_cache is None or self._daily_ribbon_cache_date != today:
            candles = self.get_candles("1d", 250)
            closes = [float(c["c"]) for c in candles] if candles else []
            self._daily_ribbon_cache = (self.calculate_ema(closes, self.EMA_FAST),
                                        self.calculate_ema(closes, self.EMA_MID),
                                        self.calculate_ema(closes, self.EMA_SLOW))
            self._daily_ribbon_cache_date = today
        return rules.daily_ribbon_confirms(*self._daily_ribbon_cache)

    def _select_setup(self, ema9, ema21, ema50, band_label=None):
        direction, band, price = self._raw_setup(ema9, ema21, ema50, band_label)
        if band is None:
            return None, None, None
        if self.regime_blocks_longs:
            return None, None, None
        if not self._daily_ribbon_confirms():
            return None, None, None
        return direction, band, price

    def _near_extreme_data(self):
        today = self._today()
        if self._near_extreme_cache is None or self._near_extreme_cache_date != today:
            candles = self.get_candles("1d", max(self.near_high_lookback, self.lh_recent + self.lh_prior) + 2)
            if not candles:
                return None
            closes_all_done = [float(c["c"]) for c in candles[:-1]]
            done = closes_all_done[-self.near_high_lookback:]
            self._lower_high_cache = rules.lower_high_stats(closes_all_done, self.lh_recent, self.lh_prior) or ()
            self._near_extreme_cache = ((done[-1], max(done), len(done)) if len(done) >= 2 else ())
            self._near_extreme_cache_date = today
        return self._near_extreme_cache

    def _near_extreme_check(self):
        if self.near_high_max_pct <= 0:
            return True, ""
        data = self._near_extreme_data()
        if not data:
            return True, ""
        last, hi, n = data
        dist = rules.near_extreme_dist(last, hi)
        return dist <= self.near_high_max_pct, f"{dist * 100:.1f}% below the {n}d closing high"

    def _limit_extension_check(self, limit_price):
        if self.limit_extension_max_pct <= 0 or not limit_price or limit_price <= 0:
            return True, ""
        data = self._near_extreme_data()
        if not data:
            return True, ""
        _last, hi, n = data
        ext = rules.limit_extension_pct(hi, limit_price)
        return ext <= self.limit_extension_max_pct, f"{n}d closing high is {ext * 100:.1f}% above the limit"

    def _lower_high_check(self):
        if not self.lower_high_filter:
            return True, ""
        if self._near_extreme_data() is None:
            return True, ""
        stats = self._lower_high_cache
        if not stats:
            return True, ""
        if not rules.lower_high_fails(stats):
            return True, ""
        return False, "lower high"

    def _entry_gates_pass(self, limit_price):
        for check in (self._near_extreme_check, lambda: self._limit_extension_check(limit_price),
                      self._lower_high_check):
            ok, detail = check()
            if not ok:
                return False, detail
        return True, ""

    def flat_reason(self, ema9, ema21, ema50, prev_close, live_price, closed_ribbon, drawdown_paused):
        """Why no entry limit is resting while flat (read-only, but it warms the daily caches exactly as the
        original's `_flat_reason` does - the first call of a day freezes those values for the whole day)."""
        if drawdown_paused:
            return "paused (daily drawdown limit)"
        if self._volume_blocked:
            return "not on the watchlist"
        if self.last_entry_fill_date == self._today():
            return "already entered today (one entry per day)"
        direction, band, limit_price = self._select_setup(ema9, ema21, ema50)
        if band is None:
            d, b, _p = self._raw_setup(ema9, ema21, ema50)
            if b is None:
                return "ribbon not stacked (live EMA9 >= EMA21 >= EMA50 needed)"
            if self.regime_blocks_longs:
                return "regime blocks longs"
            return "daily ribbon does not confirm"
        if not rules.closed_candle_confirms(prev_close, *closed_ribbon, band):
            return f"last closed candle {prev_close:.2f} is not above the {band}+ ribbon"
        ok, detail = self._entry_gates_pass(limit_price)
        if not ok:
            return detail
        if rules.would_fill_immediately(limit_price, live_price):
            return f"price {live_price:.2f} already past limit {limit_price:.2f} (would fill at market)"
        return f"all gates pass - limit {band}@{limit_price:.2f} placing"

    # ═══════════════════════ maintenance ═══════════════════════
    def _protect_partial_entry(self):
        pos = self.get_position()
        size = abs(pos.get("size") or 0)
        entry = pos.get("entry") or 0
        if size <= 0 or entry <= 0:
            if self._partial_stop_size > 0:
                self.cancel_all_trigger_orders()
                self.stop_loss_oid = None
            self._partial_stop_size = 0.0
            return
        if self._partial_stop_size > 0 and abs(size - self._partial_stop_size) <= self._partial_stop_size * 0.01:
            return
        pct = self.STOP_TIERS[-1][0]
        if self._replace_stop(size, entry * (1 - pct)):
            self._partial_stop_size = size

    def _maintain_entry_limit(self, ema9, ema21, ema50, prev_close, live_price, completed_ts,
                              closed_ribbon=(None, None, None)):
        today = self._today()
        if self.last_entry_fill_date == today:
            if self.pending_entry_oid:
                self._cancel_entry_limit()
            return
        direction, band, limit_price = self._select_setup(ema9, ema21, ema50)
        if band is None:
            if self.pending_entry_oid:
                self._cancel_entry_limit()
            return
        if not rules.closed_candle_confirms(prev_close, *closed_ribbon, band):
            if self.pending_entry_oid:
                self._cancel_entry_limit()
            return
        ok, _detail = self._entry_gates_pass(limit_price)
        if not ok:
            if self.pending_entry_oid:
                self._cancel_entry_limit()
            return
        if rules.would_fill_immediately(limit_price, live_price):
            if self.pending_entry_oid:
                self._cancel_entry_limit()
            return
        same_setup = self.pending_entry_oid and self.pending_entry_band == band
        need_replace = (not self.pending_entry_oid or not same_setup
                        or (self._now() - self.last_limit_refresh_ts) >= self.LIMIT_REFRESH_SEC)
        if not need_replace:
            return
        target = self.initial_entry_long
        add = target
        if self.pending_entry_oid and same_setup:
            already = abs(self.get_position().get("size", 0))
            add = max(0.0, target - already * limit_price)
        if self.pending_entry_oid:
            self._cancel_entry_limit()
        if add <= 0:
            return
        oid = self._place_entry_limit(limit_price, add, completed_ts, band)
        if oid:
            self.last_limit_refresh_ts = self._now()

    def _add_size_pct(self):
        pct = self.ADDITIONAL_ENTRY_SIZE_PCT - self.ADD_SIZE_STEP_PCT * self.additional_adds_done
        return max(pct, min(self.ADD_SIZE_MIN_PCT, self.ADDITIONAL_ENTRY_SIZE_PCT))

    def _maintain_additional_limit(self, ema9, ema21, ema50, prev_close, live_price, completed_ts,
                                   closed_ribbon=(None, None, None)):
        if self.initial_entry_snapshot <= 0:
            return
        today = self._today()
        if self.last_add_fill_date == today:
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        if self.additional_adds_done >= self.MAX_ADDITIONAL_ADDS:
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        if self.pyramid_interval != self.signal_interval:
            e = self._timeframe_emas(self.pyramid_interval)
            if e[0] is None:
                if self.pending_add_oid:
                    self._cancel_additional_limit()
                return
            ema9, ema21, ema50, e9d, e21d, e50d, p_prev = e
            prev_close = p_prev
            closed_ribbon = (e9d, e21d, e50d)
        add_band = self.pyramid_band_label
        direction, band, limit_price = self._select_setup(ema9, ema21, ema50, band_label=add_band)
        if band is None:
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        if not rules.closed_candle_confirms(prev_close, *closed_ribbon, add_band):
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        pos = self.get_position()
        entry = pos.get("entry", 0) or self.entry_price
        prof_pct = (live_price - entry) / entry if (entry and entry > 0 and live_price > 0) else -1.0
        if prof_pct < self.PYRAMID_MIN_PROFIT_PCT:
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        if rules.would_fill_immediately(limit_price, live_price):
            if self.pending_add_oid:
                self._cancel_additional_limit()
            return
        add_usd = self.initial_entry_snapshot * self._add_size_pct()
        need_replace = (not self.pending_add_oid or self.pending_add_band != band
                        or (self._now() - self.last_add_refresh_ts) >= self.LIMIT_REFRESH_SEC)
        if not need_replace:
            return
        if self.pending_add_oid:
            self._cancel_additional_limit()
        oid = self._place_additional_limit(limit_price, add_usd, band)
        if oid:
            self.last_add_refresh_ts = self._now()

    def _timeframe_emas(self, interval):
        candles = self.get_candles(interval, 250)
        if len(candles) < self.EMA_SLOW + 2:
            return (None,) * 7
        closes_all = [float(c["c"]) for c in candles]
        closes_done = [float(c["c"]) for c in candles[:-1]]
        return (self.calculate_ema(closes_all, self.EMA_FAST), self.calculate_ema(closes_all, self.EMA_MID),
                self.calculate_ema(closes_all, self.EMA_SLOW), self.calculate_ema(closes_done, self.EMA_FAST),
                self.calculate_ema(closes_done, self.EMA_MID), self.calculate_ema(closes_done, self.EMA_SLOW),
                float(candles[-2]["c"]))

    def _check_exit(self, c_close, ema9, ema21, ema50, closes_done=None):
        if not self.position_type:
            return False
        reason = rules.check_exit(closes_done or [], ema9, ema21, ema50, self.exit_mode, self.exit_profit_2closes,
                                  self.exit_lower_close_max_closes, self.EMA_FAST, self.EMA_MID, self.EMA_SLOW)
        if reason:
            self._market_close_position(reason=reason, current_price=c_close)
            return True
        return False

    # ═══════════════════════ the cycle ═══════════════════════
    def check_strategy(self, drawdown_paused: bool = False):
        candles = self.get_candles(self.signal_interval, 250)
        if len(candles) < self.EMA_SLOW + 2:
            return
        completed = candles[-2]
        completed_ts = int(completed["t"])
        closes_all = [float(c["c"]) for c in candles]
        closes_done = [float(c["c"]) for c in candles[:-1]]
        ema9_live = self.calculate_ema(closes_all, self.EMA_FAST)
        ema21_live = self.calculate_ema(closes_all, self.EMA_MID)
        ema50_live = self.calculate_ema(closes_all, self.EMA_SLOW)
        ema9_done = self.calculate_ema(closes_done, self.EMA_FAST)
        ema21_done = self.calculate_ema(closes_done, self.EMA_MID)
        ema50_done = self.calculate_ema(closes_done, self.EMA_SLOW)
        if ema9_live is None or ema9_done is None:
            return
        self._last_ema = (ema9_live, ema21_live, ema50_live)
        prev_close = float(completed["c"])
        c_close = prev_close

        # 1. stop-fill detection
        if self.check_stop_loss_filled():
            self.last_processed_candle_ts = completed_ts
            return
        if self._check_tiered_stops_filled():
            self.last_processed_candle_ts = completed_ts
            return

        # 2. pending entry fill
        if self.pending_entry_oid:
            filled, fill_price = self._check_entry_limit_filled()
            if filled and fill_price:
                self._on_entry_limit_filled(fill_price, ema9_live, ema21_live, ema50_live)
                self.last_processed_candle_ts = completed_ts
                return
            if self.pending_entry_oid:
                self._protect_partial_entry()

        # 3. pending add fill
        if self.pending_add_oid:
            filled, fill_price = self._check_additional_limit_filled()
            if filled and fill_price:
                self._on_additional_filled(fill_price)

        # 4. restart sync
        if self.position_type is None:
            if self._restart_sync(ema9_live, ema21_live, ema50_live, completed_ts):
                return

        # 5. defensive checks
        self._defensive_stop_check()
        self._enforce_single_orders()

        # 6. per-candle dedup
        new_candle = completed_ts > self.last_processed_candle_ts
        live_price = closes_all[-1]

        # 6.5 status (kept for its cache-warming side effects)
        if self.position_type is None and not self.pending_entry_oid:
            self.flat_reason(ema9_live, ema21_live, ema50_live, prev_close, live_price,
                             (ema9_done, ema21_done, ema50_done), drawdown_paused)

        # 7. exit
        if self.position_type and new_candle:
            if self._check_exit(c_close, ema9_done, ema21_done, ema50_done, closes_done=closes_done):
                self.last_processed_candle_ts = completed_ts
                return

        # 7.5 take-profits
        if self.position_type:
            self.check_tp_filled()
            self.check_tp2_filled()
            self._maintain_tp2(live_price)

        # 8. scale-in while open
        if self.position_type and not drawdown_paused:
            self._maintain_additional_limit(ema9_live, ema21_live, ema50_live, prev_close, live_price, completed_ts,
                                            closed_ribbon=(ema9_done, ema21_done, ema50_done))

        # 9. entry while flat
        if self.position_type is None and not drawdown_paused and not self._volume_blocked:
            self._maintain_entry_limit(ema9_live, ema21_live, ema50_live, prev_close, live_price, completed_ts,
                                       closed_ribbon=(ema9_done, ema21_done, ema50_done))

        if new_candle:
            self.last_processed_candle_ts = completed_ts
