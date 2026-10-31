"""`KiteBroker` - the Zerodha (Kite Connect) implementation of `trader.Broker` for ONE NSE stock.

Request shapes follow the official `kiteconnect` Python SDK (v5 signatures were read from its source).
Response field names (orders / positions / holdings / GTT) follow Kite's documentation and have NOT been
exercised against a live account - the sandbox this was written in has no access to Zerodha.  Treat the first
run as a dry run (see `zerodha_runner.py`).

How the original bot's Hyperliquid order model is mapped onto Kite
-----------------------------------------------------------------
 original (Hyperliquid)                         here (Zerodha, product CNC = delivery)
 ----------------------------------------      -------------------------------------------------------------
 GTC entry / add limit (rests for days)         regular LIMIT BUY, validity DAY.  It expires at 15:30; the
                                                trader already re-places a missing entry/add limit every poll
 reduce-only stop-market trigger                GTT (single) SELL: when LTP falls to the trigger, Kite places a
                                                LIMIT sell at `limit_px` (the original also uses trigger*0.95)
 reduce-only TP limit (TP1 / TP2)               GTT (single) SELL at the TP price (triggers when LTP rises to it)
 market_close                                   LIMIT SELL, validity IOC, at the 3%-through price the original uses

GTT ids are exposed as `GTT_OFFSET + id` so they cannot collide with order ids and still sort "newer = larger".
A GTT is classified as a stop (vs a take-profit) from where its trigger sits relative to the LTP it was created
at: a SELL trigger below that price is a stop, above it a take-profit.  This needs no local state, so it
survives restarts.

Position = holdings (settled + T+1) + today's CNC net buys from positions().  If your account reports T+1 shares
in both places you will see doubles - check `position()` on the first dry run (`count_t1`).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from .data import ist_now, resample_session_bars
from .trader import OrderView

GTT_OFFSET = 10 ** 18
SESSION_CLOSE = (15, 30)
SESSION_OPEN = (9, 15)
_DONE = {"COMPLETE", "CANCELLED", "REJECTED"}


class KiteUnsupported(RuntimeError):
    """A request Kite cannot serve (unknown interval or instrument)."""


def _ms(dt: datetime) -> int:
    """Epoch-milliseconds of an IST wall-clock time (only ordering matters to the trader)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _next_weekday(d: datetime) -> datetime:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


class KiteBroker:
    def __init__(self, kite, symbol: str, *, exchange: str = "NSE", product: str = "CNC",
                 now: Callable[[], datetime] = ist_now, sleep: Callable[[float], None] = time.sleep,
                 throttle_sec: float = 0.4, daily_lookback_days: int = 420, hourly_lookback_days: int = 120,
                 count_t1: bool = True):
        self.k, self.symbol, self.exchange, self.product = kite, symbol, exchange, product
        self.now, self.sleep = now, sleep
        self.throttle = throttle_sec            # Kite allows ~3 historical-data requests / second
        self.daily_days, self.hourly_days = daily_lookback_days, hourly_lookback_days
        self.count_t1 = count_t1
        self._token: Optional[int] = None
        self._last_call = 0.0

    # ───────────────────────── plumbing ─────────────────────────
    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    def token(self) -> int:
        if self._token is None:
            for ins in self.k.instruments(self.exchange):
                if ins.get("tradingsymbol") == self.symbol and ins.get("instrument_type", "EQ") == "EQ":
                    self._token = int(ins["instrument_token"])
                    break
            if self._token is None:
                raise KiteUnsupported(f"{self.key} not found in Kite's instrument list")
        return self._token

    def ltp(self) -> float:
        data = self.k.ltp(self.key)
        return float(data[self.key]["last_price"])

    def _throttle(self):
        wait = self.throttle - (time.monotonic() - self._last_call)
        if wait > 0:
            self.sleep(wait)
        self._last_call = time.monotonic()

    # ───────────────────────── candles ─────────────────────────
    def _fetch(self, kite_interval: str, days: int) -> List[dict]:
        self._throttle()
        to = self.now()
        rows = self.k.historical_data(self.token(), to - timedelta(days=days), to, kite_interval)
        out = []
        for r in rows:
            d = r["date"]
            d = d.replace(tzinfo=None) if getattr(d, "tzinfo", None) else d     # Kite stamps IST already
            out.append({"start": d, "o": float(r["open"]), "h": float(r["high"]), "l": float(r["low"]),
                        "c": float(r["close"]), "v": float(r.get("volume", 0))})
        return out

    @staticmethod
    def _session_close(day: datetime) -> datetime:
        return day.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1], second=0, microsecond=0)

    def _end_of(self, start: datetime, minutes: Optional[int]) -> datetime:
        close = self._session_close(start)
        return close if minutes is None else min(start + timedelta(minutes=minutes), close)

    def _with_forming(self, cands: List[dict], minutes: Optional[int]) -> List[dict]:
        """Guarantee the LAST candle is the one still forming - which the trader (like the original, running 24/7)
        assumes.  Kite omits it before the open and after the close, so a synthetic flat candle at the current
        price is appended; the previous candle then counts as completed."""
        if not cands:
            return []
        now, last = self.now(), cands[-1]
        if now < self._end_of(last["start"], minutes):
            return cands                                               # still forming (market open)
        px = self.ltp()
        if minutes is None:                                            # daily
            today = datetime(now.year, now.month, now.day)
            last_day = datetime(last["start"].year, last["start"].month, last["start"].day)
            if last_day < today:                                       # today's candle does not exist yet
                nxt = today if today.weekday() < 5 else _next_weekday(today)
            else:                                                      # today's is complete (after the close)
                nxt = _next_weekday(last_day)
        else:                                                          # intraday: next slot, or next session open
            slot = last["start"] + timedelta(minutes=minutes)
            if slot >= self._session_close(slot) or slot.date() != last["start"].date():
                day = _next_weekday(datetime(last["start"].year, last["start"].month, last["start"].day))
                today = datetime(now.year, now.month, now.day)
                if day < today and today.weekday() < 5:
                    day = today
                nxt = day.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1])
            else:
                nxt = slot
        cands.append({"start": nxt, "o": px, "h": px, "l": px, "c": px, "v": 0.0})
        return cands

    def candles(self, interval: str, count: int) -> List[dict]:
        iv = interval.lower()
        if iv == "1m":
            px = self.ltp()
            return [{"t": 0, "o": px, "h": px, "l": px, "c": px, "v": 0.0}]
        if iv in ("1d", "d", "24h"):
            cands, minutes = self._fetch("day", self.daily_days), None
            for c in cands:
                c["start"] = datetime(c["start"].year, c["start"].month, c["start"].day)
        elif iv in ("1h", "60m", "60minute"):
            cands, minutes = self._fetch("60minute", self.hourly_days), 60
        elif iv in ("2h", "4h"):
            hours = int(iv[:-1])
            raw = self._fetch("60minute", self.hourly_days)
            df = pd.DataFrame(raw).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["turnover"] = df["close"] * df["volume"]
            df = df.set_index("start")[["open", "high", "low", "close", "volume", "turnover"]]
            res = resample_session_bars(df, hours)
            cands = [{"start": ts.to_pydatetime(), "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume}
                     for ts, r in res.iterrows()]
            minutes = hours * 60
        else:
            raise KiteUnsupported(f"candle interval '{interval}' is not supported")
        cands = self._with_forming(cands, minutes)
        out = [{"t": _ms(c["start"]), "o": c["o"], "h": c["h"], "l": c["l"], "c": c["c"], "v": c["v"]} for c in cands]
        return out[-count:]

    # ───────────────────────── account state ─────────────────────────
    def position(self) -> dict:
        sym, ex = self.symbol, self.exchange
        held_q = held_cost = 0.0
        for h in self.k.holdings() or []:
            if h.get("tradingsymbol") == sym and h.get("exchange", ex) == ex:
                q = float(h.get("quantity", 0)) + (float(h.get("t1_quantity", 0)) if self.count_t1 else 0.0)
                held_q += q
                held_cost += q * float(h.get("average_price", 0))
        buy_q = buy_cost = sell_q = 0.0
        net = (self.k.positions() or {}).get("net", [])
        for p in net:
            if p.get("tradingsymbol") == sym and p.get("exchange", ex) == ex and p.get("product") == self.product:
                bq = float(p.get("day_buy_quantity", p.get("buy_quantity", 0)) or 0)
                bp = float(p.get("day_buy_price", p.get("buy_price", 0)) or 0)
                buy_q += bq
                buy_cost += bq * bp
                sell_q += float(p.get("day_sell_quantity", p.get("sell_quantity", 0)) or 0)
        qty = held_q + buy_q - sell_q
        gross_q = held_q + buy_q
        avg = (held_cost + buy_cost) / gross_q if gross_q > 0 else 0.0
        if qty <= 0:
            return {"size": 0.0, "entry": 0.0, "pnl": 0.0}
        return {"size": qty, "entry": avg, "pnl": (self.ltp() - avg) * qty}

    def orders(self) -> List[OrderView]:
        out: List[OrderView] = []
        for o in self.k.orders() or []:
            if o.get("tradingsymbol") != self.symbol or o.get("exchange", self.exchange) != self.exchange:
                continue
            if o.get("status") in _DONE:
                continue
            buy = o.get("transaction_type") == "BUY"
            pending = o.get("pending_quantity")
            size = float(pending if pending not in (None, 0) else o.get("quantity", 0))
            is_trig = o.get("order_type") in ("SL", "SL-M")
            out.append(OrderView(int(o["order_id"]), buy, size, float(o.get("price") or 0.0),
                                 reduce_only=not buy, is_trigger=is_trig,
                                 trigger_px=float(o.get("trigger_price") or 0.0)))
        for g in self.k.get_gtts() or []:
            cond = g.get("condition", {})
            if g.get("status") != "active" or cond.get("tradingsymbol") != self.symbol \
                    or cond.get("exchange", self.exchange) != self.exchange or not g.get("orders"):
                continue
            leg = g["orders"][0]
            trig = float(cond["trigger_values"][0])
            last = float(cond.get("last_price", trig))
            sell = leg.get("transaction_type") == "SELL"
            is_stop = (sell and trig < last) or ((not sell) and trig > last)
            out.append(OrderView(GTT_OFFSET + int(g["id"]), not sell, float(leg["quantity"]), float(leg["price"]),
                                 reduce_only=True, is_trigger=is_stop, trigger_px=trig if is_stop else 0.0))
        return out

    # ───────────────────────── order entry ─────────────────────────
    def _place(self, txn: str, qty: int, order_type: str, price: float, validity: str = "DAY") -> int:
        oid = self.k.place_order(variety="regular", exchange=self.exchange, tradingsymbol=self.symbol,
                                 transaction_type=txn, quantity=int(qty), product=self.product,
                                 order_type=order_type, price=float(price), validity=validity)
        return int(oid)

    def _gtt(self, trigger: float, qty: int, limit_px: float) -> int:
        res = self.k.place_gtt(
            trigger_type="single", tradingsymbol=self.symbol, exchange=self.exchange, trigger_values=[float(trigger)],
            last_price=self.ltp(),
            orders=[{"transaction_type": "SELL", "quantity": int(qty), "order_type": "LIMIT",
                     "product": self.product, "price": float(limit_px)}])
        return GTT_OFFSET + int(res["trigger_id"])

    def place_limit(self, is_buy: bool, size: float, price: float, reduce_only: bool = False):
        qty = int(size)
        if qty < 1:
            return None
        if is_buy and not reduce_only:                                 # entry / scale-in
            return "resting", self._place("BUY", qty, "LIMIT", price)
        if (not is_buy) and reduce_only:                               # take-profit
            if self.ltp() >= price:                                    # already through the target: a GTT could
                return "resting", self._place("SELL", qty, "LIMIT", price)      # not trigger, so sell now
            return "resting", self._gtt(price, qty, price)
        raise ValueError("a long-only trader only buys to enter and sells (reduce-only) to take profit")

    def place_stop(self, size: float, trigger_px: float, limit_px: float):
        qty = int(size)
        if qty < 1:
            return None
        last = self.ltp()
        if last <= trigger_px:                                         # already through the stop: exit now, with a
            px = round(min(limit_px, last * 0.97) / 0.05) * 0.05       # limit BELOW the market so it is marketable
            return "resting", self._place("SELL", qty, "LIMIT", round(px, 2))
        return "resting", self._gtt(trigger_px, qty, limit_px)

    def cancel(self, oid: int) -> None:
        if oid >= GTT_OFFSET:
            self.k.delete_gtt(oid - GTT_OFFSET)
        else:
            self.k.cancel_order(variety="regular", order_id=str(oid))

    def market_close(self, size: float, ref_price: float) -> bool:
        oid = self._place("SELL", int(size), "LIMIT", ref_price, validity="IOC")
        for _ in range(4):                                             # order status updates asynchronously
            for o in self.k.orders() or []:
                if int(o["order_id"]) == oid:
                    if o.get("status") == "COMPLETE":
                        return True
                    if o.get("status") in ("CANCELLED", "REJECTED"):
                        return False
            self.sleep(0.5)
        return False
