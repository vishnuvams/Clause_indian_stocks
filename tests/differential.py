"""Run the ORIGINAL trader and `indian_stocks.trader.Trader` side by side on identical fake exchanges."""
from __future__ import annotations

import collections
import random
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

from indian_stocks.config import StrategyConfig
from indian_stocks.trader import Broker, OrderView, Trader, Venue
from tests import oracle

DAY_S = 86_400
T0 = 1_700_006_400.0                      # a UTC midnight (2023-11-15 00:00:00)


# ───────────────────────────── broker adapter over the fake exchange ─────────────────────────────

class FakeBroker:
    def __init__(self, hl: oracle.FakeHL):
        self.hl = hl

    def candles(self, interval, count):
        return self.hl.candles_snapshot({"interval": interval})[-count:]

    def position(self):
        h = self.hl
        return {"size": h.pos_size, "entry": h.pos_entry if h.pos_size else 0.0,
                "pnl": (h.price - h.pos_entry) * h.pos_size}

    def orders(self):
        return [OrderView(o.oid, o.is_buy, o.sz, o.px, o.reduce_only, o.is_trigger, o.trigger_px)
                for o in self.hl.orders.values()]

    def _res(self, r):
        s = r["response"]["data"]["statuses"][0]
        if "resting" in s:
            return "resting", s["resting"]["oid"]
        return "filled", None

    def place_limit(self, is_buy, size, price, reduce_only=False):
        return self._res(self.hl.order(self.hl.coin, is_buy, size, price, {"limit": {"tif": "Gtc"}},
                                       reduce_only=reduce_only))

    def place_stop(self, size, trigger_px, limit_px):
        return self._res(self.hl.order(self.hl.coin, False, size, limit_px,
                                       {"trigger": {"triggerPx": trigger_px, "isMarket": True, "tpsl": "sl"}},
                                       reduce_only=True))

    def cancel(self, oid):
        self.hl.cancel(self.hl.coin, oid)

    def market_close(self, size, ref_price):
        return self.hl.market_close(self.hl.coin, size, ref_price)["status"] == "ok"


CFG_FIELDS = set(StrategyConfig.__dataclass_fields__)


def cfg_from_kwargs(kw: dict) -> StrategyConfig:
    """StrategyConfig from the very kwargs the original trader was built with (same names, USD -> INR numbers)."""
    d = {k: v for k, v in kw.items() if k in CFG_FIELDS}
    if "initial_entry_usd_long" in kw:
        d["initial_entry_inr_long"] = kw["initial_entry_usd_long"]
    d.setdefault("initial_entry_inr_long", 900_000.0)
    if "stop_tiers" in d:
        d["stop_tiers"] = list(d["stop_tiers"])
    return StrategyConfig(**d)


# ───────────────────────────── scenario generation ─────────────────────────────

def gen_bars(rng: random.Random, n: int, p0=100.0, regimes=((0.007, 0.022),), switch_every=70,
             gap_prob=0.0, gap_size=(0.08, 0.2)) -> List[Tuple[float, float, float, float]]:
    bars, p, k = [], p0, 0
    for i in range(n):
        drift, vol = regimes[(i // switch_every) % len(regimes)]
        o = p * (1 + rng.gauss(0, vol / 5))
        if gap_prob and rng.random() < gap_prob:          # overnight gap (down or up) - exercises gap-through-stop
            o *= 1 - rng.uniform(*gap_size) * (1 if rng.random() < 0.7 else -1)
        c = o * (1 + rng.gauss(drift, vol))
        h = max(o, c) * (1 + abs(rng.gauss(0, vol / 2)))
        l = min(o, c) * (1 - abs(rng.gauss(0, vol / 2)))
        bars.append((o, h, l, c))
        p = c
    return bars


def candle(day_idx: int, o: float, h: float, l: float, c: float) -> dict:
    return {"t": int((T0 + day_idx * DAY_S) * 1000), "o": o, "h": h, "l": l, "c": c, "v": 1_000_000}


# ───────────────────────────── the pair ─────────────────────────────

@dataclass
class World:
    hl: oracle.FakeHL
    clock: oracle.Clock
    trader: object
    name: str


STATE_FIELDS = ("position_type", "additional_adds_done", "partial_tp_done", "tp2_done", "last_entry_fill_date",
                "last_add_fill_date", "pending_entry_side", "pending_entry_band", "pending_add_side",
                "pending_add_band", "_tiers_active")
STATE_FLOATS = ("position_size", "entry_price", "emergency_stop_price", "pending_entry_price", "pending_add_price",
                "tp_price", "tp_size", "tp2_price", "tp2_size")


def state_of(tr) -> dict:
    d = {k: getattr(tr, k) for k in STATE_FIELDS}
    for k in STATE_FLOATS:
        v = getattr(tr, k)
        d[k] = None if v is None else round(float(v), 4)
    d["tiers"] = sorted((round(t["pct"], 4), round(t["frac"], 4)) for t in tr._tier_stop_oids)
    return d


class Pair:
    def __init__(self, mod, hist: List[tuple], *, over: Optional[dict] = None, rs_direction="LONG",
                 initial_position: Optional[Tuple[float, float]] = None, partial_frac: float = 1.0,
                 setup: Optional[Callable[["World"], None]] = None):
        self.mod = mod
        self.worlds: List[World] = []
        kw = oracle.stocks_trader_kwargs(mod)
        kw.update(over or {})
        cfg = cfg_from_kwargs(kw)
        for name in ("original", "port"):
            clock = oracle.Clock(T0 + len(hist) * DAY_S + 300)
            oracle.bind_clock(mod, clock)
            hl = oracle.FakeHL(price=hist[-1][3], partial_frac=partial_frac)
            hl.candles["1d"] = [candle(i, *b) for i, b in enumerate(hist)]
            if initial_position:
                hl.pos_size, hl.pos_entry = initial_position
            orig = oracle.make_original_trader(mod, hl, over, rs_direction=rs_direction)
            if name == "original":
                tr = orig
            else:
                venue = Venue(fmt_price=orig._format_price, round_size=orig.round_size)
                tr = Trader(FakeBroker(hl), cfg, venue, now=clock.time,
                            today=lambda c=clock: __import__("datetime").datetime.fromtimestamp(
                                c.t, tz=__import__("datetime").timezone.utc).strftime("%Y-%m-%d"))
            w = World(hl, clock, tr, name)
            if setup:
                setup(w)                     # e.g. pre-existing stop orders for restart-sync scenarios
            self.worlds.append(w)
        self.day = len(hist)
        self.hook: Optional[Callable[[int, "Pair"], None]] = None
        self.drawdown_paused = False
        self.events = collections.Counter()
        self._prev_state = state_of(self.worlds[0].trader)
        self._prev_closes = 0

    def set_all(self, **attrs):
        for w in self.worlds:
            for k, v in attrs.items():
                setattr(w.trader, k, v)

    def set_regime(self, blocks_longs: bool):
        """Benchmark below its EMA21: the original expresses it as regime_allowed="SHORT" (which blocks LONG),
        the long-only port as a boolean."""
        orig, port = self.worlds[0].trader, self.worlds[1].trader
        orig.regime_allowed = "SHORT" if blocks_longs else "LONG"
        port.regime_blocks_longs = blocks_longs

    def _each(self, fn):
        for w in self.worlds:
            oracle.bind_clock(self.mod, w.clock)         # the original reads the module-level clock
            fn(w)

    def poll(self, t_in_day: float):
        def run(w):
            w.clock.t = T0 + self.day * DAY_S + t_in_day
            w.trader.check_strategy(drawdown_paused=self.drawdown_paused)
        self._each(run)
        self._track()

    def _track(self):
        """Count lifecycle events seen in the ORIGINAL world (so tests can assert what a scenario exercised)."""
        w = self.worlds[0]
        s, p = state_of(w.trader), self._prev_state
        closes = sum(1 for e in w.hl.log if e[0] == "market_close")
        ev = self.events
        if p["position_type"] is None and s["position_type"]:
            ev["position_opened"] += 1
        if s["additional_adds_done"] > p["additional_adds_done"]:
            ev["add_filled"] += 1
        if s["partial_tp_done"] and not p["partial_tp_done"]:
            ev["tp1_filled"] += 1
        if s["tp2_done"] and not p["tp2_done"]:
            ev["tp2_filled"] += 1
        if p["position_type"] and not s["position_type"]:
            ev["exit_by_market_close" if closes > self._prev_closes else "closed_by_stop_or_tp"] += 1
        if p["position_type"] and s["position_type"] and len(s["tiers"]) < len(p["tiers"]):
            ev["tier_stop_fired"] += 1
        if p["position_type"] and s["position_type"] and p["position_type"] != s["position_type"]:
            ev["flip"] += 1
        if s["pending_entry_side"] and not p["pending_entry_side"]:
            ev["entry_limit_placed"] += 1
        if s["pending_add_side"] and not p["pending_add_side"]:
            ev["add_limit_placed"] += 1
        if s["tp2_price"] and not p["tp2_price"]:
            ev["tp2_limit_placed"] += 1
        self._prev_state, self._prev_closes = s, closes

    def mismatch(self) -> Optional[str]:
        a, b = self.worlds
        sa, sb = a.hl.snapshot(), b.hl.snapshot()
        if sa != sb:
            return f"exchange state differs\n  original: {sa}\n  port:     {sb}"
        ta, tb = state_of(a.trader), state_of(b.trader)
        if ta != tb:
            diff = {k: (ta[k], tb[k]) for k in ta if ta[k] != tb[k]}
            return f"trader state differs {diff}"
        return None

    def play_day(self, bar, rng: random.Random, polls: int = 4) -> Optional[str]:
        if self.hook:
            self.hook(self.day, self)
        o, h, l, c = bar
        self._each(lambda w: w.hl.move_to(o))
        self._each(lambda w: w.hl.candles["1d"].append(candle(self.day, o, o, o, o)))
        path = [o] + ([h, l] if rng.random() < 0.5 else [l, h]) + [c]
        times = [300, 6 * 3600, 12 * 3600, 18 * 3600][:polls]
        hi, lo = o, o
        for price, t in zip(path, times):
            def step(w, price=price):
                w.hl.move_to(price)
                cd = w.hl.candles["1d"][-1]
                cd["h"], cd["l"], cd["c"] = max(cd["h"], price), min(cd["l"], price), price
            self._each(step)
            self.poll(t)
            m = self.mismatch()
            if m:
                return f"day {self.day} t={t}s price={price:.2f}: {m}"
        self.day += 1
        return None

    def run(self, bars, rng, polls=4) -> Optional[str]:
        for b in bars:
            m = self.play_day(b, rng, polls)
            if m:
                return m
        return None


# ───────────────────────────── intraday (1h / 4h / 1d candles, 30-minute polls) ─────────────────────────────

HOUR_S = 3_600


def hourly_path(rng: random.Random, bar) -> List[float]:
    """25 prices (hour 0..24) that run o -> (h,l or l,h) -> c linearly between 4 waypoints."""
    o, h, l, c = bar
    mid = [h, l] if rng.random() < 0.5 else [l, h]
    cuts = sorted(rng.sample(range(2, 23), 2))
    pts = [(0, o), (cuts[0], mid[0]), (cuts[1], mid[1]), (24, c)]
    out = []
    for (h0, p0), (h1, p1) in zip(pts, pts[1:]):
        for k in range(h0, h1):
            out.append(p0 + (p1 - p0) * (k - h0) / (h1 - h0))
    out.append(c)
    return out


def _ms(day: int, hour: int) -> int:
    return int((T0 + day * DAY_S + hour * HOUR_S) * 1000)


class IntradayPair(Pair):
    """Same comparison, but the exchange serves 1h, 4h and 1d candles and the traders poll every 30 minutes."""

    def __init__(self, mod, hist_days: List[tuple], rng: random.Random, **kw):
        # history = hourly candles built from daily bars, so every interval is mutually consistent
        self._rng = rng
        h1, h4, d1 = [], [], []
        for d, bar in enumerate(hist_days):
            path = hourly_path(rng, bar)
            for k in range(24):
                o, c = path[k], path[k + 1]
                h1.append({"t": _ms(d, k), "o": o, "h": max(o, c), "l": min(o, c), "c": c, "v": 1})
            day = h1[-24:]
            d1.append({"t": _ms(d, 0), "o": day[0]["o"], "h": max(x["h"] for x in day),
                       "l": min(x["l"] for x in day), "c": day[-1]["c"], "v": 1})
            for j in range(0, 24, 4):
                blk = day[j:j + 4]
                h4.append({"t": _ms(d, j), "o": blk[0]["o"], "h": max(x["h"] for x in blk),
                           "l": min(x["l"] for x in blk), "c": blk[-1]["c"], "v": 1})
        super().__init__(mod, hist_days, **kw)
        for w in self.worlds:
            w.hl.candles = {"1h": [dict(c) for c in h1], "4h": [dict(c) for c in h4], "1d": [dict(c) for c in d1]}
            w.hl.price = h1[-1]["c"]
        # the history above ENDS with completed candles; the first simulated hour opens a new one
        self._frac_polls = (5 * 60, 35 * 60)

    def _roll(self, w, day, hour, price):
        """Open a new in-progress candle for every interval that starts at (day, hour)."""
        cs = w.hl.candles
        cs["1h"].append({"t": _ms(day, hour), "o": price, "h": price, "l": price, "c": price, "v": 1})
        if hour % 4 == 0:
            cs["4h"].append({"t": _ms(day, hour), "o": price, "h": price, "l": price, "c": price, "v": 1})
        if hour == 0:
            cs["1d"].append({"t": _ms(day, 0), "o": price, "h": price, "l": price, "c": price, "v": 1})

    def _touch_candles(self, w, price):
        for itv in ("1h", "4h", "1d"):
            cd = w.hl.candles[itv][-1]
            cd["h"], cd["l"], cd["c"] = max(cd["h"], price), min(cd["l"], price), price

    def play_day(self, bar, rng=None, polls=None) -> Optional[str]:
        if self.hook:
            self.hook(self.day, self)
        path = hourly_path(self._rng, bar)
        for hour in range(24):
            p0, p1 = path[hour], path[hour + 1]
            self._each(lambda w: w.hl.move_to(p0))
            self._each(lambda w: self._roll(w, self.day, hour, p0))
            self._each(lambda w: self._touch_candles(w, p0))
            for off in self._frac_polls:
                price = p0 + (p1 - p0) * off / HOUR_S
                self._each(lambda w, price=price: (w.hl.move_to(price), self._touch_candles(w, price)))
                t_in_day = hour * HOUR_S + off
                self.poll(t_in_day)
                m = self.mismatch()
                if m:
                    return f"day {self.day} {hour:02d}:{off // 60:02d} price={price:.2f}: {m}"
        self.day += 1
        return None
