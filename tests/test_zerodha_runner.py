import dataclasses
from datetime import datetime, timezone

import pytest

from indian_stocks.config import StrategyConfig
from indian_stocks.zerodha_runner import CONFIRM_ENV, CONFIRM_VALUE, DryRunBroker, ZerodhaRunner
from tests import differential as D
from tests import oracle
from tests.fake_kite import FakeKite
from tests.helpers import trend

RELAXED = dataclasses.replace(
    StrategyConfig(), min_turnover_inr=0.0, adr_min_pct=0.0, rs_min_pct_long=0.0, near_high_max_pct=0.0,
    recent_rs_filter=False, ema21_rising_filter=False, lower_high_filter=False, limit_extension_max_pct=0.0)


def world(pos=None, n=131):
    """A steadily rising stock: EMA ribbon stacked, price above its EMA9 => an entry limit should be rested."""
    hl = oracle.FakeHL(coin="NSE:AAA", price=0.0)
    df = trend(n, g=0.01)
    hl.candles["1d"] = [D.candle(i, r.open, r.high, r.low, r.close) for i, (_, r) in enumerate(df.iterrows())]
    hl.price = hl.candles["1d"][-1]["c"]
    if pos:
        hl.pos_size, hl.pos_entry = pos
    day = datetime.fromtimestamp(D.T0 + (n - 1) * D.DAY_S, tz=timezone.utc).replace(tzinfo=None, hour=11, minute=0)
    return hl, FakeKite(hl), (lambda: day)


def runner(kite, now, **kw):
    logs = []
    r = ZerodhaRunner(kite, kw.pop("cfg", RELAXED), ["AAA"], now=now, sleep=lambda s: None, log=logs.append, **kw)
    return r, logs


def test_dry_run_reads_real_data_but_sends_nothing():
    hl, kite, now = world()
    r, logs = runner(kite, now)
    r.cycle()
    assert r.watchlist == ["AAA"] and "AAA" in r.traders
    sent = [c for c in kite.calls if c[0] in ("place_order", "place_gtt", "cancel_order", "delete_gtt")]
    assert sent == [] and hl.orders == {}
    assert any("would place BUY limit" in m for m in logs)
    assert isinstance(r.traders["AAA"].b, DryRunBroker)


def test_live_needs_the_environment_confirmation(monkeypatch):
    _, kite, now = world()
    monkeypatch.delenv(CONFIRM_ENV, raising=False)
    with pytest.raises(PermissionError):
        runner(kite, now, live=True)
    monkeypatch.setenv(CONFIRM_ENV, "yes")
    with pytest.raises(PermissionError):
        runner(kite, now, live=True)


def test_live_places_a_real_day_limit_buy(monkeypatch):
    monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
    hl, kite, now = world()
    r, _ = runner(kite, now, live=True)
    r.cycle()
    orders = [c[1] for c in kite.calls if c[0] == "place_order"]
    assert len(orders) == 1
    o = orders[0]
    assert (o["transaction_type"], o["order_type"], o["product"], o["validity"]) == ("BUY", "LIMIT", "CNC", "DAY")
    assert o["price"] < hl.price                                     # a pullback limit rests BELOW the market
    assert round(o["price"] / 0.05) * 0.05 == pytest.approx(o["price"])      # on the NSE tick
    assert len(hl.orders) == 1


def test_a_stock_you_already_hold_is_not_taken_over_by_default():
    hl, kite, now = world(pos=(40, 120.0))
    r, logs = runner(kite, now)
    r.cycle()
    assert "AAA" not in r.traders and any("you already hold it" in m for m in logs)
    assert [c for c in kite.calls if c[0] in ("place_order", "place_gtt")] == []


def test_adopt_existing_manages_the_position_and_would_protect_it():
    hl, kite, now = world(pos=(40, 120.0))
    r, logs = runner(kite, now, adopt_existing=True)
    r.cycle()
    assert "AAA" in r.traders and r.traders["AAA"].position_type == "LONG"
    assert any("would place STOP (GTT)" in m for m in logs)
    assert not any(c[0] in ("place_order", "place_gtt") for c in kite.calls)


def test_a_failure_in_one_stock_does_not_stop_the_cycle():
    hl, kite, now = world()
    r, logs = runner(kite, now)
    r.refresh_watchlist()
    r.traders["AAA"].check_strategy = lambda **k: (_ for _ in ()).throw(RuntimeError("boom"))
    r.cycle()
    assert any("error RuntimeError: boom" in m for m in logs)


def test_missing_benchmark_falls_back_to_raw_returns_and_says_so():
    _, kite, now = world()
    r, logs = runner(kite, now)
    r.refresh_watchlist()
    assert any("no benchmark" in m for m in logs) and r.watchlist == ["AAA"]
