"""Zerodha runner - the thin "bot" layer: watchlist from Kite data, one `Trader` per stock, poll loop.

SAFETY
    * DRY RUN by default: reads your real Kite data and logs the orders it WOULD place; sends nothing.
    * Live trading needs BOTH `live=True` AND the environment variable ZERODHA_CONFIRM_LIVE=I_UNDERSTAND.
    * It only ever manages symbols from the universe you pass.  A stock you ALREADY hold is skipped unless
      `adopt_existing=True`, because `Trader` treats the whole holding as its own position (it would put a stop
      on it and may sell it).
    * Never run it on a Kite account with money you are not prepared to lose: this layer has been tested only
      against a fake Kite.

Credentials: KITE_API_KEY and KITE_ACCESS_TOKEN (the token expires every morning - generate it with Kite's
normal login flow; this code does not automate the login).
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from .config import StrategyConfig
from .data import ist_now
from .frames import Frame
from .kite_broker import KiteBroker, KiteUnsupported
from .selection import Candidate, build_watchlist, regime_allowed
from .trader import OrderView, Trader, Venue

CONFIRM_ENV, CONFIRM_VALUE = "ZERODHA_CONFIRM_LIVE", "I_UNDERSTAND"


class DryRunBroker:
    """Reads go to the real broker; writes are only logged.  Keeps a virtual order book so the trader's own
    bookkeeping (pending orders, stops) behaves sensibly between polls."""

    def __init__(self, real: KiteBroker, log: Callable[[str], None]):
        self.real, self.log, self.virtual, self._id = real, log, {}, 0

    def _new(self, ov: OrderView) -> int:
        self._id += 1
        ov.oid = -self._id                      # negative: can never collide with a real id
        self.virtual[ov.oid] = ov
        return ov.oid

    def candles(self, interval, count):
        return self.real.candles(interval, count)

    def position(self):
        return self.real.position()

    def orders(self):
        return self.real.orders() + list(self.virtual.values())

    def place_limit(self, is_buy, size, price, reduce_only=False):
        kind = "TAKE-PROFIT" if reduce_only else ("BUY limit" if is_buy else "SELL limit")
        self.log(f"[dry-run] {self.real.symbol}: would place {kind} {int(size)} @ {price:.2f}")
        if not is_buy and not reduce_only:
            raise KiteUnsupported("opening a short is not possible in delivery")
        return "resting", self._new(OrderView(0, is_buy, size, price, reduce_only))

    def place_stop(self, is_buy, size, trigger_px, limit_px):
        self.log(f"[dry-run] {self.real.symbol}: would place STOP (GTT) {int(size)} trigger {trigger_px:.2f} "
                 f"limit {limit_px:.2f}")
        return "resting", self._new(OrderView(0, is_buy, size, limit_px, True, True, trigger_px))

    def cancel(self, oid):
        if oid in self.virtual:
            del self.virtual[oid]
        else:
            self.log(f"[dry-run] {self.real.symbol}: would cancel order {oid}")

    def market_close(self, is_long, size, ref_price):
        self.log(f"[dry-run] {self.real.symbol}: would SELL {int(size)} (IOC limit {ref_price:.2f}) to close")
        return True


class ZerodhaRunner:
    def __init__(self, kite, cfg: StrategyConfig, symbols: List[str], *, live: bool = False,
                 benchmark: str = "NIFTY 50", fallback_benchmark: str = "NIFTYBEES", adopt_existing: bool = False,
                 refresh_minutes: float = 120, now: Callable[[], datetime] = ist_now,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
                 blacklist=()):
        if live and os.environ.get(CONFIRM_ENV) != CONFIRM_VALUE:
            raise PermissionError(f"live trading needs {CONFIRM_ENV}={CONFIRM_VALUE} in the environment")
        self.kite, self.cfg, self.symbols, self.live = kite, cfg, [s.upper() for s in symbols], live
        self.benchmark, self.fallback, self.adopt = benchmark, fallback_benchmark, adopt_existing
        self.refresh_min, self.now, self.sleep, self.log = refresh_minutes, now, sleep, log
        self.blacklist = list(blacklist)
        self.traders: Dict[str, Trader] = {}
        self._last_refresh: Optional[datetime] = None
        self.regime: Optional[str] = None
        self.watchlist: List[str] = []

    # ---- plumbing ----
    def _kb(self, sym: str) -> KiteBroker:
        return KiteBroker(self.kite, sym, now=self.now, sleep=self.sleep)

    def _new_trader(self, sym: str) -> Trader:
        kb = self._kb(sym)
        broker = kb if self.live else DryRunBroker(kb, self.log)
        tr = Trader(broker, self.cfg, Venue(tick=self.cfg.tick_size), now=time.time,
                    today=lambda: self.now().strftime("%Y-%m-%d"), log=self.log)
        tr._kb = kb
        return tr

    def _benchmark_frame(self):
        for name in (self.benchmark, self.fallback):
            try:
                return Frame.from_candles(self._kb(name).candles("1d", 250))[0]
            except Exception as exc:                          # index not in the instrument dump, no data, ...
                self.log(f"benchmark '{name}' unavailable ({exc})")
        self.log("no benchmark: relative strength falls back to raw returns")
        return None

    # ---- watchlist (EMAPullbackBot._refresh_stocks + _apply_selection) ----
    def refresh_watchlist(self):
        B = self._benchmark_frame()
        kb_i = (len(B) - 1) if B is not None else -1
        self.regime = regime_allowed(B, kb_i, self.cfg, self.regime)
        cands = []
        for sym in self.symbols:
            try:
                D, x = Frame.from_candles(self._kb(sym).candles("1d", 250))
            except Exception as exc:
                self.log(f"{sym}: no data ({exc})")
                continue
            if len(D):
                cands.append(Candidate(sym, D, len(D) - 1, x))
        selected, dropped = build_watchlist(cands, B, kb_i, self.cfg, self.blacklist)
        self.watchlist = [s.symbol for s in selected]
        self.log(f"watchlist: {len(selected)} of {len(cands)} stocks: {', '.join(self.watchlist[:25])}")
        for tr in self.traders.values():
            tr.in_scanner, tr._volume_blocked = False, True
        for s in selected:
            tr = self.traders.get(s.symbol)
            if tr is None:
                if not self.adopt and self._kb(s.symbol).position()["size"] != 0:
                    self.log(f"{s.symbol}: you already hold it - skipped (adopt_existing=False)")
                    continue
                tr = self.traders[s.symbol] = self._new_trader(s.symbol)
            tr.in_scanner, tr._volume_blocked, tr.rs_direction = True, False, s.direction
        self._last_refresh = self.now()

    # ---- one pass over every trader (EMAPullbackBot.check_all) ----
    def cycle(self):
        if self._last_refresh is None or self.now() - self._last_refresh >= timedelta(minutes=self.refresh_min):
            self.refresh_watchlist()
        for sym in list(self.traders):
            tr = self.traders[sym]
            tr.regime_allowed = self.regime if self.cfg.regime_filter_enabled else None
            if tr._volume_blocked and tr.position_type is None and not tr.pending_entry_oid:
                continue
            try:
                tr.check_strategy()
            except KiteUnsupported as exc:
                self.log(f"{sym}: {exc}")
            except Exception as exc:                          # one stock must never stop the others
                self.log(f"{sym}: error {type(exc).__name__}: {exc}")
        for sym in [s for s, t in self.traders.items()
                    if t.position_type is None and not t.in_scanner and not t.pending_entry_oid]:
            del self.traders[sym]

    def run_forever(self, poll_minutes: float = 30):
        while True:
            n = self.now()
            if n.weekday() < 5 and (9, 15) <= (n.hour, n.minute) < (15, 30):
                self.cycle()
            self.sleep(poll_minutes * 60)


def kite_from_env():
    from kiteconnect import KiteConnect
    key, token = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_ACCESS_TOKEN")
    if not key or not token:
        raise SystemExit("set KITE_API_KEY and KITE_ACCESS_TOKEN (the token expires every morning)")
    k = KiteConnect(api_key=key)
    k.set_access_token(token)
    return k
