from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from indian_stocks import data, universe

BHAV = """\
SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
RELIANCE, EQ, 31-Oct-2025, 100.00, 101.00, 105.00, 99.00, 104.00, 104.50, 103.00, 1000000, 1030.00, 5000, 600000, 60.00
RELIANCE, BE, 31-Oct-2025, 10.00, 10.00, 10.50, 9.90, 10.20, 10.30, 10.1, 100, 1.01, 5, 60, 60.00
TCS, EQ, 31-Oct-2025, 3000.00, -, -, -, -, -, -, 0, 0, 0, 0, 0
INFY, EQ, 31-Oct-2025, 1500.00, 1510.00, 1520.00, 1490.00, 1505.00, 1500.00, 1502.00, 2000000, 30040.00, 9000, 900000, 45.00
"""


def test_parse_bhavcopy_handles_leading_spaces_series_and_missing_values():
    df = data.parse_bhavcopy(BHAV)
    assert sorted(df["symbol"]) == ["INFY", "RELIANCE"]                  # BE series and the all-'-' row dropped
    r = df[df["symbol"] == "RELIANCE"].iloc[0]
    assert r["close"] == 104.5 and r["volume"] == 1_000_000
    assert r["turnover"] == pytest.approx(1030.0 * 1e5)                   # lakh -> INR
    assert r["date"] == pd.Timestamp("2025-10-31")


def test_parse_bhavcopy_rejects_unknown_format():
    with pytest.raises(ValueError):
        data.parse_bhavcopy("A,B\n1,2\n")


class FakeResp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


class FakeSession:
    def __init__(self, days):
        self.days, self.calls = days, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        if "nseindia.com/products" not in url and "nsearchives" not in url:
            return FakeResp(200, "")
        key = url.rsplit("_", 1)[1].split(".")[0]                         # ddmmyyyy
        return FakeResp(200, self.days[key]) if key in self.days else FakeResp(404)


def test_nse_provider_builds_frames_caches_files_and_remembers_holidays(tmp_path):
    day = lambda d, close: BHAV.replace("31-Oct-2025", d).replace("104.50", str(close))
    sess = FakeSession({"29102025": day("29-Oct-2025", 101), "31102025": day("31-Oct-2025", 104.5)})
    prov = data.NSEBhavcopyProvider(cache_dir=str(tmp_path), session=sess, pause_sec=0)
    out = prov.fetch(["RELIANCE", "NOPE"], "2025-10-27", "2025-10-31")
    assert list(out) == ["RELIANCE"]
    df = out["RELIANCE"]
    assert list(df.columns) == data.OHLCV and len(df) == 2                 # 30-Oct was a "holiday" (404)
    assert df["close"].tolist() == [101.0, 104.5]
    n_calls = len(sess.calls)
    again = prov.fetch(["RELIANCE"], "2025-10-27", "2025-10-31")           # served from the cache: no new downloads
    assert len(sess.calls) == n_calls and len(again["RELIANCE"]) == 2
    assert prov.turnover_snapshot(date(2025, 10, 31))["RELIANCE"] == pytest.approx(1030e5)


def test_nse_provider_is_daily_only(tmp_path):
    with pytest.raises(ValueError):
        data.NSEBhavcopyProvider(cache_dir=str(tmp_path), session=FakeSession({})).fetch(["X"], "2025-01-01", "2025-01-02", "1h")


def test_normalize_ohlcv_cleans_and_converts_timezone():
    idx = pd.DatetimeIndex(["2025-01-02 03:45", "2025-01-02 04:45", "2025-01-02 04:45", "2025-01-02 05:45"], tz="UTC")
    raw = pd.DataFrame({"Open": [1, 2, 3, 4], "High": [2, 3, 4, 1], "Low": [1, 1, 1, 2], "Close": [1.5, 2.5, 3.5, 3],
                        "Volume": [10, 20, 30, 40]}, index=idx)
    out = data.normalize_ohlcv(raw)
    assert out.index.tz is None and out.index[0] == pd.Timestamp("2025-01-02 09:15")        # UTC -> IST
    assert len(out) == 2                                                 # duplicate dropped, high<low row dropped
    assert out["turnover"].iloc[0] == pytest.approx(1.5 * 10)


def test_resample_session_bars_never_straddles_sessions():
    rows = []
    for d in ("2025-01-02", "2025-01-03"):
        for k in range(7):                                               # 09:15 ... 15:15 hourly
            ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(hours=k)
            rows.append((ts, 100 + k, 101 + k, 99 + k, 100.5 + k, 10, 1000))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"]).set_index("ts")
    out = data.resample_session_bars(df, 4)
    assert len(out) == 4                                                 # [4h + 3h] per session
    first = out.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (100, 104, 99, 103.5, 40)
    assert out.index[1] == pd.Timestamp("2025-01-02 13:15") and out.iloc[1]["volume"] == 30
    assert out.index[2].date() == date(2025, 1, 3)


def test_drop_incomplete_last_bar_daily_and_hourly():
    df = pd.DataFrame({"open": [1, 1], "high": [2, 2], "low": [1, 1], "close": [1, 1], "volume": [1, 1],
                       "turnover": [1, 1]}, index=pd.DatetimeIndex(["2025-10-30", "2025-10-31"]))
    assert len(data.drop_incomplete_last_bar(df, "1d", now=datetime(2025, 10, 31, 11, 0))) == 1     # market open
    assert len(data.drop_incomplete_last_bar(df, "1d", now=datetime(2025, 10, 31, 15, 31))) == 2    # closed
    assert len(data.drop_incomplete_last_bar(df, "1d", now=datetime(2025, 11, 1, 9, 0))) == 2       # next morning
    h = df.copy()
    h.index = pd.DatetimeIndex(["2025-10-31 13:15", "2025-10-31 14:15"])
    assert len(data.drop_incomplete_last_bar(h, "1h", now=datetime(2025, 10, 31, 14, 40))) == 1
    assert len(data.drop_incomplete_last_bar(h, "1h", now=datetime(2025, 10, 31, 15, 20))) == 2


def test_yahoo_provider_normalises_batches_and_uses_its_cache(tmp_path, monkeypatch):
    idx = pd.bdate_range("2025-01-01", periods=40)
    def frame(base):
        return pd.DataFrame({"Open": base, "High": base + 1, "Low": base - 1, "Close": base + 0.5,
                             "Volume": 1000}, index=idx)
    cols = pd.MultiIndex.from_product([["AAA.NS", "^NSEI"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.concat({"AAA.NS": frame(100.0), "^NSEI": frame(20000.0)}, axis=1)
    raw.columns = cols
    calls = []
    prov = data.YahooProvider(cache_dir=str(tmp_path))
    monkeypatch.setattr(prov, "_download", lambda tickers, s, e, i: calls.append(tickers) or raw)
    out = prov.fetch(["AAA", "^NSEI", "MISSING"], "2025-01-01", "2025-02-25")
    assert calls[0] == ["AAA.NS", "^NSEI", "MISSING.NS"]                 # index keeps its ticker, stocks get .NS
    assert set(out) == {"AAA", "^NSEI"} and out["AAA"]["close"].iloc[0] == 100.5
    prov.fetch(["AAA", "^NSEI"], "2025-01-01", "2025-02-25")             # second time: from the CSV cache
    assert len(calls) == 1


def test_csv_provider(tmp_path):
    pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "Open": [1, 2], "High": [2, 3], "Low": [1, 1],
                  "Close": [1.5, 2.5], "Volume": [100, 200]}).to_csv(tmp_path / "AAA.csv", index=False)
    out = data.CsvProvider(str(tmp_path)).fetch(["AAA", "NONE"], "2025-01-02", "2025-01-02")
    assert list(out) == ["AAA"] and len(out["AAA"]) == 1 and out["AAA"]["turnover"].iloc[0] == 500


# ───────────────────────── universe ─────────────────────────

@pytest.mark.parametrize("symbol,name,series,ok", [
    ("RELIANCE", "Reliance Industries Limited", "EQ", True),
    ("M&M", "Mahindra & Mahindra Limited", "EQ", True),
    ("BAJAJ-AUTO", "Bajaj Auto Limited", "EQ", True),
    ("GOLDBEES", "Nippon India ETF Gold BeES", "EQ", False),
    ("NIFTYBEES", "Nippon India ETF Nifty 50 BeES", "EQ", False),
    ("LIQUIDCASE", "Zerodha Nifty Liquid ETF", "EQ", False),
    ("SOMEINVIT", "Some Infrastructure Investment Trust", "EQ", False),
    ("EMBASSY", "Embassy Office Parks REIT", "EQ", False),
    ("XYZ", "XYZ Limited", "BE", False),
    ("XYZ", "XYZ Limited", "SM", False),
    ("ABC-RE", "ABC Rights", "EQ", False),
    ("bad symbol", "", "EQ", False),
])
def test_is_stock(symbol, name, series, ok):
    assert universe.is_stock(symbol, name, series) is ok


def test_parse_equity_l_and_index_formats_and_filter():
    equity_l = ("SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE\n"
                "RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10\n"
                "GOLDBEES,Nippon India ETF Gold BeES,EQ,08-MAR-2007,1,1,INF204KB17I5,1\n")
    index_csv = ("Company Name,Industry,Symbol,Series,ISIN Code\n"
                 "Infosys Ltd.,IT,INFY,EQ,INE009A01021\nSome Fund,Fin,FUNDX,BE,X\n")
    a = universe.filter_stocks(universe.parse_universe_csv(equity_l))
    b = universe.filter_stocks(universe.parse_universe_csv(index_csv), blacklist=["infy"])
    assert a["symbol"].tolist() == ["RELIANCE"] and b.empty


def test_load_universe_file_and_offline_fallback(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("Symbol,Series\nAAA,EQ\nBBB,EQ\n")
    assert universe.load_universe(f"file:{p}")["symbol"].tolist() == ["AAA", "BBB"]
    fb = universe.load_universe("nifty50", cache_dir=str(tmp_path / "c"), offline=True)
    assert len(fb) >= 25 and "RELIANCE" in fb["symbol"].tolist() and "M&M" in fb["symbol"].tolist()
    with pytest.raises(ValueError):
        universe.load_universe("nifty9000")


def test_select_by_turnover_orders_and_floors():
    out = universe.select_by_turnover({"A": 5e7, "B": 9e7, "C": 1e6, "D": 7e7}, 4.5e7, blacklist=["d"])
    assert out == [("B", 9e7), ("A", 5e7)]
