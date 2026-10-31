"""The ORIGINAL bot as an oracle.

`load_original()` imports `ema_9_21_50_stocks_crypto_1day_1hour.py` with the Hyperliquid SDK stubbed out and
its clock (`time`, `datetime`) replaced by a controllable one.  `FakeHL` is a small exchange (order book,
position, stop-market triggers, fills on price moves) that implements the exact `info` / `exchange` calls the
original makes, so the real `EMAPullbackTrader.check_strategy()` / `EMAPullbackBot._refresh_stocks()` can be
driven on synthetic data and their behaviour compared with `indian_stocks`.

Set EMA_BOT_PATH to the original file (default below).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

PATH = os.environ.get("EMA_BOT_PATH", "/home/user/clude_crypto/ema_9_21_50_stocks_crypto_1day_1hour.py")
DAY = 86_400


def original_available() -> bool:
    return os.path.exists(PATH)


# ───────────────────────────── clock ─────────────────────────────

class Clock:
    def __init__(self, t: float):
        self.t = float(t)

    def time(self) -> float:
        return self.t

    def sleep(self, _s) -> None:        # never actually sleeps, never advances time
        return None


class _TimeModule:
    def __init__(self, clock: Clock):
        self._c = clock

    def time(self):
        return self._c.time()

    def sleep(self, s):
        return self._c.sleep(s)

    def __getattr__(self, name):        # anything else (strftime, ...) from the real module
        import time as real
        return getattr(real, name)


def _fake_datetime(clock: Clock):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(clock.t, tz=tz or timezone.utc)
    return FakeDT


# ───────────────────────────── loader ─────────────────────────────

_CACHED = None


def load_original():
    """Import the original once (SDK stubbed).  Returns the module; call `bind_clock` before using it."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    for name in ["hyperliquid", "hyperliquid.info", "hyperliquid.exchange", "hyperliquid.utils",
                 "hyperliquid.utils.constants", "eth_account"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["hyperliquid.info"].Info = object
    sys.modules["hyperliquid.exchange"].Exchange = object
    sys.modules["hyperliquid.utils"].constants = sys.modules["hyperliquid.utils.constants"]
    spec = importlib.util.spec_from_file_location("orig_ema_bot", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHED = mod
    return mod


def bind_clock(mod, clock: Clock):
    mod.time = _TimeModule(clock)
    mod.datetime = _fake_datetime(clock)


def stocks_trader_kwargs(mod) -> dict:
    """The kwargs `EMAPullbackBot` passes to each trader for market="stocks" (see `_build_bot`)."""
    v = mod.VENUE_CFG["stocks"]
    return dict(
        dex_name=v["dex_name"],
        additional_entry_size_pct=v["additional_entry_size_pct"],
        additional_entry_size_step_pct=v.get("additional_entry_size_step_pct", 0.0),
        additional_entry_size_min_pct=v.get("additional_entry_size_min_pct", 0.0),
        max_additional_adds=v["max_additional_adds"],
        pyramid_min_profit_pct=mod.PYRAMID_MIN_PROFIT_PCT,
        pyramid_interval=v["pyramid_interval"],
        ema_fast=mod.EMA_FAST_PERIOD, ema_mid=mod.EMA_MID_PERIOD, ema_slow=mod.EMA_SLOW_PERIOD,
        stop_loss_pct_ema9=v["stop_loss_pct_ema9"], stop_loss_pct_ema21=v["stop_loss_pct_ema21"],
        stop_loss_pct_ema50=v["stop_loss_pct_ema50"], stop_tiers=v.get("stop_tiers"),
        limit_refresh_hours=mod.LIMIT_REFRESH_HOURS,
        tp_profit_pct=mod.TP_PROFIT_PCT, tp_size_pct=mod.TP_SIZE_PCT, tp_stop_to_breakeven=mod.TP_STOP_TO_BREAKEVEN,
        enable_shorts=mod.ENABLE_SHORTS, candle_interval_hours=mod.CANDLE_INTERVAL_HOURS,
        signal_interval=v["signal_interval"], exit_mode=v["exit_mode"], exit_profit_2closes=v["exit_profit_2closes"],
        entry_ema=v["entry_ema"], entry_ema_short=v.get("entry_ema_short"),
        pyramid_ema=v["pyramid_ema"], pyramid_ema_short=v.get("pyramid_ema_short"),
        stop_to_new_avg_after_add=v["stop_to_new_avg_after_add"], daily_ribbon_confirm=v["daily_ribbon_confirm"],
        near_high_max_pct=v.get("near_high_max_pct", 0.0), near_high_lookback=mod.NEAR_HIGH_LOOKBACK_DAYS,
        limit_extension_max_pct=v.get("limit_extension_max_pct", 0.0),
        lower_high_filter=v.get("lower_high_filter", False),
        lower_high_recent_days=mod.LOWER_HIGH_RECENT_DAYS, lower_high_prior_days=mod.LOWER_HIGH_PRIOR_DAYS,
        ema50_tp_mult=mod.EMA50_TP_MULT, ema50_tp_size_pct=mod.EMA50_TP_SIZE_PCT,
        ema50_tp_adr_period=mod.ADR_PERIOD, exit_lower_close_max_closes=mod.EXIT_LOWER_CLOSE_MAX_CLOSES,
    )


# ───────────────────────────── fake exchange ─────────────────────────────

@dataclass
class Order:
    oid: int
    coin: str
    is_buy: bool
    sz: float
    px: float                      # limit price (for triggers: the limit/slippage price)
    reduce_only: bool = False
    trigger_px: float = 0.0        # > 0 => stop-market trigger order

    @property
    def is_trigger(self) -> bool:
        return self.trigger_px > 0


class FakeHL:
    """One symbol, one account.  Implements the subset of Info/Exchange the original uses."""

    def __init__(self, coin="xyz:AAA", sz_decimals=0, balance=10_000_000.0, price=100.0, partial_frac=1.0):
        self.partial_frac = partial_frac        # < 1: a resting non-reduce limit fills only this fraction per touch
        self.coin = coin
        self.sz_decimals = sz_decimals
        self.balance = balance
        self.price = price
        self.candles: Dict[str, List[dict]] = {"1d": []}
        self.orders: Dict[int, Order] = {}
        self.pos_size = 0.0
        self.pos_entry = 0.0
        self._oid = 1000
        self.log: List[tuple] = []
        self.fills: List[tuple] = []

    # ---- helpers -------------------------------------------------
    def _new_oid(self) -> int:
        self._oid += 1
        return self._oid

    def snapshot(self) -> dict:
        """Comparable view of the exchange state (order ids stripped)."""
        orders = sorted((o.is_buy, round(o.sz, 6), round(o.px, 4), o.reduce_only, round(o.trigger_px, 4))
                        for o in self.orders.values())
        return {"pos": (round(self.pos_size, 6), round(self.pos_entry, 4)), "orders": orders}

    # ---- Info API ------------------------------------------------
    def candles_snapshot(self, req):                  # `req` is the single parameter => api_params == ['req']
        itv = req["interval"]
        if itv == "1m":
            return [{"t": 0, "o": self.price, "h": self.price, "l": self.price, "c": self.price, "v": 1}]
        return list(self.candles.get(itv, []))

    def post(self, path, payload):
        t = payload.get("type")
        if t == "meta":
            return {"universe": [{"name": self.coin, "szDecimals": self.sz_decimals,
                                  "onlyIsolated": True, "maxLeverage": 5}]}
        if t == "clearinghouseState":
            ap = []
            if self.pos_size != 0:
                ap.append({"position": {"coin": self.coin, "szi": str(self.pos_size), "entryPx": str(self.pos_entry),
                                        "unrealizedPnl": str((self.price - self.pos_entry) * self.pos_size)}})
            return {"assetPositions": ap, "marginSummary": {"accountValue": str(self.balance)}}
        if t == "spotClearinghouseState":
            return {"balances": [{"coin": "USDC", "token": 0, "total": str(self.balance)}]}
        if t == "openOrders":
            return [self._basic(o) for o in self.orders.values()]
        if t == "frontendOpenOrders":
            return [self._front(o) for o in self.orders.values()]
        raise NotImplementedError(f"FakeHL.post type {t}")

    def _basic(self, o: Order) -> dict:
        d = {"coin": self.coin, "oid": o.oid, "side": "B" if o.is_buy else "A", "sz": str(o.sz), "limitPx": str(o.px)}
        return d

    def _front(self, o: Order) -> dict:
        d = self._basic(o)
        d.update({"orderType": "Stop Market" if o.is_trigger else "Limit",
                  "triggerPx": str(o.trigger_px) if o.is_trigger else "0.0",
                  "isTrigger": bool(o.is_trigger), "reduceOnly": bool(o.reduce_only),
                  "isPositionTpsl": False})
        return d

    # ---- Exchange API --------------------------------------------
    def order(self, coin, is_buy, sz, limit_px, order_type, reduce_only=False):
        self.log.append(("order", coin, is_buy, sz, limit_px, order_type, reduce_only))
        if "trigger" in order_type:
            o = Order(self._new_oid(), coin, is_buy, float(sz), float(limit_px), True,
                      float(order_type["trigger"]["triggerPx"]))
            self.orders[o.oid] = o
            return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": o.oid}}]}}}
        o = Order(self._new_oid(), coin, is_buy, float(sz), float(limit_px), bool(reduce_only))
        marketable = (is_buy and o.px >= self.price) or ((not is_buy) and o.px <= self.price)
        if marketable:
            self._fill(o, self.price)
            return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"oid": o.oid}}]}}}
        self.orders[o.oid] = o
        return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": o.oid}}]}}}

    def cancel(self, coin, oid):
        self.log.append(("cancel", oid))
        self.orders.pop(int(oid), None)
        return {"status": "ok"}

    def market_close(self, coin, sz, px):
        self.log.append(("market_close", sz))
        if self.pos_size != 0:
            self._apply(-self.pos_size if abs(sz) >= abs(self.pos_size) else -sz if self.pos_size > 0 else sz, self.price)
        return {"status": "ok"}

    def market_open(self, coin, is_buy, sz, px):
        self.log.append(("market_open", is_buy, sz))
        self._apply(sz if is_buy else -sz, self.price)
        return {"status": "ok"}

    def update_leverage(self, *a, **k):
        return {"status": "ok"}

    # ---- position + fills ----------------------------------------
    def _apply(self, delta: float, px: float):
        """Signed position change at px."""
        new = self.pos_size + delta
        if self.pos_size == 0 or (self.pos_size > 0) == (delta > 0):
            tot = abs(self.pos_size) + abs(delta)
            self.pos_entry = (abs(self.pos_size) * self.pos_entry + abs(delta) * px) / tot if tot else 0.0
        elif abs(new) < 1e-12:
            self.pos_entry = 0.0
        elif (new > 0) != (self.pos_size > 0):          # flipped
            self.pos_entry = px
        self.pos_size = 0.0 if abs(new) < 1e-9 else new
        self.fills.append((delta, px))

    def _fill(self, o: Order, px: float, partial_ok: bool = False):
        q = o.sz
        if partial_ok and not o.reduce_only and not o.is_trigger and self.partial_frac < 1.0:
            q = max(1.0, round(o.sz * self.partial_frac))
        if o.reduce_only:
            if self.pos_size == 0 or (self.pos_size > 0) == o.is_buy:
                return
            q = min(q, abs(self.pos_size))
        self._apply(q if o.is_buy else -q, px)
        o.sz -= q

    def move_to(self, new_price: float):
        """Price moves linearly from the current price to `new_price`; resting orders it crosses fill."""
        p0, p1 = self.price, float(new_price)
        lo, hi = min(p0, p1), max(p0, p1)
        for o in sorted(self.orders.values(), key=lambda x: x.oid):
            if o.oid not in self.orders:
                continue
            if o.is_trigger:
                hit = (o.is_buy and hi >= o.trigger_px) or ((not o.is_buy) and lo <= o.trigger_px)
                if not hit:
                    continue
                gap = (o.is_buy and p0 >= o.trigger_px) or ((not o.is_buy) and p0 <= o.trigger_px)
                self._fill(o, p0 if gap else o.trigger_px)
            else:
                hit = (o.is_buy and lo <= o.px) or ((not o.is_buy) and hi >= o.px)
                if not hit:
                    continue
                self._fill(o, o.px, partial_ok=True)
            if o.sz <= 1e-9 or o.reduce_only and self.pos_size == 0:
                self.orders.pop(o.oid, None)
        # reduce-only orders larger than the remaining position are clamped (HL behaviour)
        for o in list(self.orders.values()):
            if o.reduce_only and self.pos_size != 0 and o.sz > abs(self.pos_size) + 1e-9:
                o.sz = abs(self.pos_size)
        self.price = p1


def make_original_trader(mod, hl: FakeHL, kwargs: Optional[dict] = None, *, rs_direction="LONG", symbol="AAA",
                         initial_long=900_000.0, initial_short=225_000.0):
    kw = stocks_trader_kwargs(mod)
    if kwargs:
        kw.update(kwargs)
    assert not kw.get("enable_shorts"), "the port is long-only: the oracle must run the original with shorts off"
    initial_long = kw.pop("initial_entry_usd_long", initial_long)
    initial_short = kw.pop("initial_entry_usd_short", initial_short)
    tr = mod.EMAPullbackTrader(
        symbol=symbol, leverage=1, info_client=hl, exchange_client=hl, address="0xabc",
        trade_logger=None, telegram=None,
        initial_entry_usd_long=initial_long, initial_entry_usd_short=initial_short, **kw)
    tr.rs_direction = rs_direction
    tr.in_scanner = True
    return tr
