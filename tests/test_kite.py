"""KiteBroker against a fake Kite: request shapes, GTT stop/TP classification, candle synthesis around the
session, long-only guard - and the real `Trader` driven through it, compared with the same Trader on the plain
fake broker."""
import random
import types
from datetime import datetime, timedelta, timezone

import pytest

from indian_stocks.config import StrategyConfig
from indian_stocks.kite_broker import GTT_OFFSET, KiteBroker, KiteUnsupported
from indian_stocks.trader import Trader, Venue
from tests import differential as D
from tests import oracle
from tests.fake_kite import IST, FakeKite

SYMBOL = "AAA"


def market(price=100.0, pos=None):
    hl = oracle.FakeHL(coin=f"NSE:{SYMBOL}", price=price)
    if pos:
        hl.pos_size, hl.pos_entry = pos
    return hl


def broker(hl, now=None, **kw):
    k = FakeKite(hl)
    now = now or (lambda: datetime(2025, 3, 10, 11, 0))
    return KiteBroker(k, SYMBOL, now=now, sleep=lambda s: None, throttle_sec=0, **kw), k


# ───────────────────────── request shapes ─────────────────────────

def test_entry_and_add_are_day_limit_buys_in_delivery():
    hl = market()
    b, k = broker(hl)
    assert b.place_limit(True, 18.0, 98.55)[0] == "resting"
    call = [c for c in k.calls if c[0] == "place_order"][0][1]
    assert call == dict(variety="regular", exchange="NSE", tradingsymbol="AAA", transaction_type="BUY", quantity=18,
                        product="CNC", order_type="LIMIT", price=98.55, validity="DAY")
    ov = b.orders()
    assert [(o.is_buy, o.size, o.price, o.reduce_only, o.is_trigger) for o in ov] == [(True, 18.0, 98.55, False, False)]


def test_stop_is_a_single_gtt_below_the_price_and_is_seen_as_a_trigger():
    hl = market(100.0)
    b, k = broker(hl)
    kind, oid = b.place_stop(10, 90.0, 85.5)
    assert kind == "resting" and oid >= GTT_OFFSET
    g = [c for c in k.calls if c[0] == "place_gtt"][0][1]
    assert g["trigger_type"] == "single" and g["trigger_values"] == [90.0] and g["last_price"] == 100.0
    assert g["orders"] == [{"transaction_type": "SELL", "quantity": 10, "order_type": "LIMIT", "product": "CNC",
                            "price": 85.5}]
    [o] = b.orders()
    assert (o.oid, o.is_buy, o.size, o.is_trigger, o.trigger_px, o.reduce_only) == (oid, False, 10.0, True, 90.0, True)


def test_take_profit_is_a_gtt_above_the_price_and_is_not_a_trigger():
    hl = market(100.0)
    b, k = broker(hl)
    kind, oid = b.place_limit(False, 5, 120.0, reduce_only=True)
    g = [c for c in k.calls if c[0] == "place_gtt"][0][1]
    assert g["trigger_values"] == [120.0] and g["orders"][0]["price"] == 120.0
    [o] = b.orders()
    assert (o.is_trigger, o.reduce_only, o.price, o.size) == (False, True, 120.0, 5.0)


def test_target_already_reached_sells_with_a_day_order_instead_of_an_impossible_gtt():
    hl = market(130.0, pos=(10, 100.0))
    b, k = broker(hl)
    b.place_limit(False, 3, 120.0, reduce_only=True)
    assert not any(c[0] == "place_gtt" for c in k.calls)
    assert [c[1]["transaction_type"] for c in k.calls if c[0] == "place_order"] == ["SELL"]
    assert hl.pos_size == 7                                         # marketable: it filled


def test_stop_already_breached_exits_now():
    hl = market(85.0, pos=(10, 100.0))
    b, k = broker(hl)
    b.place_stop(10, 90.0, 85.5)
    assert not any(c[0] == "place_gtt" for c in k.calls) and hl.pos_size == 0


def test_cancel_routes_gtt_ids_to_delete_gtt_and_order_ids_to_cancel_order():
    hl = market()
    b, k = broker(hl)
    _, o1 = b.place_limit(True, 5, 95.0)
    _, g1 = b.place_stop(5, 90.0, 85.5)
    b.cancel(g1)
    b.cancel(o1)
    assert [c[0] for c in k.calls if c[0] in ("delete_gtt", "cancel_order")] == ["delete_gtt", "cancel_order"]
    assert b.orders() == []


def test_only_the_long_orders_the_trader_uses_are_accepted():
    """Buy limit = entry/add, reduce-only sell = take-profit.  Anything else is a programming error, not an order."""
    b, k = broker(market())
    with pytest.raises(ValueError):
        b.place_limit(False, 5, 105.0)                              # an opening sell
    with pytest.raises(ValueError):
        b.place_limit(True, 5, 95.0, reduce_only=True)              # a reduce-only buy
    assert [c for c in k.calls if c[0] in ("place_order", "place_gtt")] == []


def test_market_close_is_an_ioc_limit_and_reports_whether_it_filled():
    hl = market(100.0, pos=(10, 90.0))
    b, k = broker(hl)
    assert b.market_close(10, 97.0) is True
    assert [c[1]["validity"] for c in k.calls if c[0] == "place_order"] == ["IOC"] and hl.pos_size == 0
    hl2 = market(100.0, pos=(10, 90.0))
    b2, _ = broker(hl2)
    assert b2.market_close(10, 101.0) is False and hl2.pos_size == 10      # IOC above the market: not filled, and it says so


def test_position_from_holdings_and_todays_buys():
    hl = market(110.0, pos=(10, 100.0))
    b, _ = broker(hl)
    assert b.position() == {"size": 10.0, "entry": 100.0, "pnl": pytest.approx(100.0)}
    # shares bought today appear in positions(); sold-today shares reduce the quantity
    k = FakeKite(hl)
    k.positions = lambda: {"net": [{"tradingsymbol": "AAA", "exchange": "NSE", "product": "CNC",
                                    "day_buy_quantity": 5, "day_buy_price": 120.0, "day_sell_quantity": 3}]}
    b2 = KiteBroker(k, SYMBOL, now=lambda: datetime(2025, 3, 10, 11, 0), sleep=lambda s: None, throttle_sec=0)
    p = b2.position()
    assert p["size"] == 12.0 and p["entry"] == pytest.approx((10 * 100 + 5 * 120) / 15)
    assert broker(market())[0].position() == {"size": 0.0, "entry": 0.0, "pnl": 0.0}


# ───────────────────────── candles: a "forming" candle must always be last ─────────────────────────

def day_rows(dates, base=100.0):
    return [{"date": datetime(d.year, d.month, d.day, tzinfo=IST), "open": base, "high": base + 2, "low": base - 2,
             "close": base + i * 0.1, "volume": 1000} for i, d in enumerate(dates)]


def kite_with_rows(rows, ltp=150.0):
    k = types.SimpleNamespace()
    k.instruments = lambda ex=None: [{"tradingsymbol": SYMBOL, "instrument_type": "EQ", "instrument_token": 7}]
    k.historical_data = lambda token, f, t, interval: list(rows)
    k.ltp = lambda *a: {f"NSE:{SYMBOL}": {"last_price": ltp}}
    return k


def daily(now, rows, ltp=150.0):
    return KiteBroker(kite_with_rows(rows, ltp), SYMBOL, now=lambda: now, sleep=lambda s: None, throttle_sec=0)


WEEKDAYS = [datetime(2025, 3, 3) + timedelta(days=i) for i in range(5)]           # Mon 3 .. Fri 7


def test_daily_during_the_session_keeps_kites_own_partial_candle_as_forming():
    rows = day_rows(WEEKDAYS + [datetime(2025, 3, 10)])                              # Monday's candle exists
    cs = daily(datetime(2025, 3, 10, 11, 0), rows).candles("1d", 250)
    assert len(cs) == 6 and cs[-1]["c"] == pytest.approx(100.5)


def test_daily_before_the_open_appends_a_flat_candle_at_the_last_price():
    rows = day_rows(WEEKDAYS)                                                        # last = Friday
    cs = daily(datetime(2025, 3, 10, 8, 30), rows, ltp=150.0).candles("1d", 250)     # Monday pre-open
    assert len(cs) == 6 and (cs[-1]["o"], cs[-1]["h"], cs[-1]["l"], cs[-1]["c"]) == (150.0, 150.0, 150.0, 150.0)
    assert cs[-1]["t"] > cs[-2]["t"]


def test_daily_after_the_close_rolls_to_the_next_trading_day():
    rows = day_rows(WEEKDAYS)                                                        # Friday's candle is complete
    cs = daily(datetime(2025, 3, 7, 16, 0), rows).candles("1d", 250)
    assert len(cs) == 6 and cs[-1]["c"] == 150.0
    nxt = datetime.fromtimestamp(cs[-1]["t"] / 1000, tz=timezone.utc)
    assert nxt.weekday() == 0                                                        # Monday, skipping the weekend


def test_daily_on_a_weekend_is_flat_and_forming():
    rows = day_rows(WEEKDAYS)
    cs = daily(datetime(2025, 3, 8, 12, 0), rows).candles("1d", 250)                  # Saturday
    assert len(cs) == 6 and cs[-1]["c"] == 150.0


def test_count_limits_the_returned_candles_and_unknown_intervals_are_refused():
    rows = day_rows(WEEKDAYS * 4)
    b = daily(datetime(2025, 3, 10, 11, 0), rows)
    assert len(b.candles("1d", 5)) == 5
    with pytest.raises(KiteUnsupported):
        b.candles("15m", 5)
    assert b.candles("1m", 1)[0]["c"] == 150.0


def hourly_rows(day: datetime, n):
    return [{"date": (day + timedelta(hours=9, minutes=15 + 60 * k)).replace(tzinfo=IST), "open": 100 + k,
             "high": 101 + k, "low": 99 + k, "close": 100.5 + k, "volume": 100} for k in range(n)]


def test_hourly_mid_session_keeps_the_partial_candle_and_after_close_starts_the_next_session():
    day = datetime(2025, 3, 10)
    rows = hourly_rows(day, 7)                                                       # 09:15 ... 15:15
    mid = KiteBroker(kite_with_rows(rows[:3]), SYMBOL, now=lambda: datetime(2025, 3, 10, 11, 40), sleep=lambda s: None,
                     throttle_sec=0).candles("1h", 250)
    assert len(mid) == 3                                                             # 11:15 candle still forming
    after = KiteBroker(kite_with_rows(rows), SYMBOL, now=lambda: datetime(2025, 3, 10, 15, 45), sleep=lambda s: None,
                       throttle_sec=0).candles("1h", 250)
    assert len(after) == 8 and after[-1]["c"] == 150.0
    nxt = datetime.fromtimestamp(after[-1]["t"] / 1000, tz=timezone.utc)
    assert (nxt.day, nxt.hour, nxt.minute) == (11, 9, 15)                            # tomorrow 09:15


def test_four_hour_candles_are_built_from_hourly_without_crossing_sessions():
    rows = hourly_rows(datetime(2025, 3, 10), 7) + hourly_rows(datetime(2025, 3, 11), 3)
    b = KiteBroker(kite_with_rows(rows), SYMBOL, now=lambda: datetime(2025, 3, 11, 11, 40), sleep=lambda s: None,
                   throttle_sec=0)
    cs = b.candles("4h", 250)
    assert len(cs) == 3                                  # Mon 09:15-13:15 and 13:15-15:30, Tue 09:15- (forming)
    assert cs[0]["h"] == max(r["high"] for r in rows[:4]) and cs[1]["o"] == rows[4]["open"]


# ───────────────────────── the real Trader through KiteBroker ─────────────────────────

class NSEFakeBroker(D.FakeBroker):
    """The plain fake broker with Kite's one behavioural difference modelled: a stop that is already through the
    market exits immediately (instead of resting)."""

    def place_stop(self, size, trigger_px, limit_px):
        if self.hl.price <= trigger_px:
            return self.place_limit(False, size, limit_px, reduce_only=True)
        return super().place_stop(size, trigger_px, limit_px)


def trader_world(kind, hist, cfg, clock):
    hl = oracle.FakeHL(coin=f"NSE:{SYMBOL}", price=hist[-1][3])
    hl.candles["1d"] = [D.candle(i, *b) for i, b in enumerate(hist)]
    if kind == "kite":
        kite = FakeKite(hl)
        b = KiteBroker(kite, SYMBOL, now=lambda: datetime.fromtimestamp(clock.t, tz=timezone.utc).replace(
            tzinfo=None, hour=11, minute=0, second=0), sleep=lambda s: None, throttle_sec=0)
    else:
        kite, b = None, NSEFakeBroker(hl)
    tr = Trader(b, cfg, Venue(tick=0.05), now=clock.time,
                today=lambda: datetime.fromtimestamp(clock.t, tz=timezone.utc).strftime("%Y-%m-%d"))
    return D.World(hl, clock, tr, kind), kite


class KiteVsPlain(D.Pair):
    def __init__(self, hist, cfg, expire_day_orders=False):
        self.mod = types.SimpleNamespace()
        self.worlds, self.kites = [], []
        for kind in ("plain", "kite"):
            clock = oracle.Clock(D.T0 + len(hist) * D.DAY_S + 300)
            w, kite = trader_world(kind, hist, cfg, clock)
            self.worlds.append(w)
            self.kites.append(kite)
        self.day, self.hook, self.drawdown_paused = len(hist), None, False
        self.events = D.collections.Counter()
        self._prev_state, self._prev_closes = D.state_of(self.worlds[0].trader), 0
        self.expire = expire_day_orders

    def play_day(self, bar, rng, polls=4):
        m = super().play_day(bar, rng, polls)
        if self.expire and self.kites[1] is not None:
            self.kites[1].expire_day_orders()                            # 15:30: regular orders lapse
            self.worlds[0].hl.orders = {k: v for k, v in self.worlds[0].hl.orders.items()
                                        if v.reduce_only or v.is_trigger}
        return m


NSE_OVER = dict(initial_entry_inr_long=900_000.0, tick_size=0.05)


def kite_cfg(**over):
    return StrategyConfig(**{**NSE_OVER, **over})


def run_pair(seed, cfg, expire=False, n=170):
    rng = random.Random(seed)
    bars = D.gen_bars(rng, 130 + n, regimes=((0.007, 0.022), (-0.004, 0.02)), switch_every=80)
    pair = KiteVsPlain(bars[:130], cfg, expire_day_orders=expire)
    return pair, pair.run(bars[130:], rng, polls=4)


def test_trader_behaves_identically_through_the_kite_adapter():
    total = D.collections.Counter()
    for seed in range(8):
        pair, msg = run_pair(seed, kite_cfg())
        assert msg is None, f"seed {seed}: {msg}"
        total.update(pair.events)
    for ev in ("position_opened", "tp1_filled", "tp2_filled", "add_filled", "exit_by_market_close"):
        assert total[ev] > 0, f"never exercised {ev}: {dict(total)}"


def test_day_orders_lapsing_every_afternoon_changes_nothing_because_stops_and_targets_are_gtt():
    total = D.collections.Counter()
    for seed in range(6):
        pair, msg = run_pair(seed, kite_cfg(), expire=True)
        assert msg is None, f"seed {seed}: {msg}"
        total.update(pair.events)
    assert total["position_opened"] > 0 and total["tp1_filled"] > 0
