import json
import math

import numpy as np
import pandas as pd
import pytest

from indian_stocks import cli
from indian_stocks.config import ExecConfig, StrategyConfig
from indian_stocks.engine import Engine
from tests.helpers import flat, make_df, random_walk, trend
from tests.test_engine import N, T, Tape, tick


def hourly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """7 NSE hourly bars/day (09:15 ... 15:15).  Path o -> h -> l -> c so the day's extremes are reached."""
    rows = []
    for ts, r in df.iterrows():
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        path = [o, o + (h - o) * 0.5, h, (h + l) / 2, l, l + (c - l) * 0.6, c]       # 7 knots -> 6 steps... + close
        opens = [o] + path[1:]
        for k in range(7):
            bo = opens[k]
            bc = path[k + 1] if k < 6 else c
            hi = max(bo, bc, h if k == 2 else 0)
            lo = min(bo, bc, l if k == 4 else 1e18)
            hi = max(hi, bo, bc)
            lo = min(lo, bo, bc)
            start = ts + pd.Timedelta(hours=9, minutes=15 + 60 * k)
            rows.append((start, bo, hi, lo, bc, r["volume"] / 7, r["turnover"] / 7))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"]).set_index("ts")


def test_hourly_mode_enters_on_the_pullback_day_with_a_daily_signal():
    t = Tape()
    t.pullback(T)
    daily = t.df()
    hourly = hourly_from_daily(daily)
    exe = ExecConfig(capital=5_000_000, trade_start="2021-03-01")
    res = Engine({"AAA": daily}, StrategyConfig(), exe, flat(N), hourly={"AAA": hourly}).run()
    entries = [f for f in res.fills if f.action == "ENTRY"]
    assert len(entries) == 1
    assert pd.Timestamp(entries[0].ts * 10**6).date() == daily.index[T].date()
    assert entries[0].units == math.floor(StrategyConfig().initial_entry_inr_long / entries[0].price)
    # exits/stops are evaluated per hourly bar but the signal stays DAILY -> still one exit decision per day
    assert res.equity[-1] > 0


def test_hourly_mode_rejects_sub_daily_signal_without_hourly_data():
    with pytest.raises(ValueError):
        Engine({"AAA": trend(120)}, StrategyConfig(signal_interval="4h"), ExecConfig(), flat(120))


def test_hourly_mode_with_4h_signal_runs_and_keeps_invariants():
    daily = {f"S{s}": random_walk(260, s, drift=0.001, vol=0.02) for s in (1, 2, 3)}
    hourly = {k: hourly_from_daily(v) for k, v in daily.items()}
    import dataclasses
    cfg = dataclasses.replace(StrategyConfig(), signal_interval="4h", pyramid_interval="4h", rs_enabled=False,
                              adr_min_pct=0.0, min_turnover_inr=0.0, near_high_max_pct=0.0, recent_rs_filter=False,
                              ema21_rising_filter=False, lower_high_filter=False, limit_extension_max_pct=0.0,
                              initial_entry_inr_long=300_000)
    res = Engine(daily, cfg, ExecConfig(capital=3_000_000, trade_start="2021-06-01"), flat(260), hourly=hourly).run()
    assert all(f.units >= 1 for f in res.fills)
    assert len(res.ts) == len(set(res.ts)) and res.ts == sorted(res.ts)


# ───────────────────────────── CLI end to end (CSV provider, no network) ─────────────────────────────

def _write_csv_book(folder, end=None, n=500):
    start = "2021-01-04"
    if end is not None:
        start = str(pd.bdate_range(end=end, periods=n)[0].date())
    g = 0.008 + 0.012 * np.sin(np.arange(n) / 9.0)
    closes = 100 * np.cumprod(1 + g)
    opens = np.concatenate([[100], closes[:-1]])
    hi, lo = np.maximum(opens, closes) * 1.025, np.minimum(opens, closes) * 0.975
    for name, scale in (("AAA", 1.0), ("BBB", 0.6)):
        df = make_df((opens * scale).tolist(), (hi * scale).tolist(), (lo * scale).tolist(), (closes * scale).tolist(),
                     volume=5_000_000, start=start)
        out = df[["open", "high", "low", "close", "volume"]].copy()
        out.index.name = "date"
        out.to_csv(folder / f"{name}.csv")
    b = flat(n, start=start)
    b[["open", "high", "low", "close", "volume"]].rename_axis("date").to_csv(folder / "NIFTYBEES.csv")


def test_cli_backtest_writes_all_outputs(tmp_path):
    _write_csv_book(tmp_path)
    out = tmp_path / "res"
    cli.main(["backtest", "--provider", "csv", "--data-dir", str(tmp_path), "--symbols", "AAA", "BBB",
              "--start", "2021-09-01", "--end", "2022-12-30", "--out", str(out), "--capital", "5000000"])
    for f in ("summary.json", "trades.csv", "fills.csv", "equity.csv", "watchlist_days.csv"):
        assert (out / f).exists()
    s = json.loads((out / "summary.json").read_text())
    assert s["closed_trades"] + s["open_positions_at_end"] >= 1
    assert s["watchlist_symbols"] == 2
    eq = pd.read_csv(out / "equity.csv")
    assert eq["equity"].iloc[0] == pytest.approx(5_000_000) and len(eq) > 200


def test_cli_scan_lists_the_watchlist(tmp_path, capsys):
    today = cli._ist_today()
    _write_csv_book(tmp_path, end=today - pd.Timedelta(days=1) if today.weekday() else today - pd.Timedelta(days=3))
    out = tmp_path / "scan.csv"
    cli.main(["scan", "--provider", "csv", "--data-dir", str(tmp_path), "--symbols", "AAA", "BBB", "--out", str(out)])
    txt = capsys.readouterr().out
    assert "stocks on the watchlist" in txt
    df = pd.read_csv(out)
    assert set(df["symbol"]) == {"AAA", "BBB"} and {"selected", "why", "entry_limit", "stop", "tp1"} <= set(df.columns)


def test_cli_universe_offline(capsys):
    cli.main(["universe", "--universe", "nifty50", "--offline", "--cache-dir", "/nonexistent-cache"])
    assert "RELIANCE" in capsys.readouterr().out
