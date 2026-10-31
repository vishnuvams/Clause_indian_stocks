"""Market-data providers.  Every provider returns {SYMBOL: DataFrame}.

DataFrame contract
    index   : tz-naive DatetimeIndex (daily: midnight of the trading day; intraday: IST wall-clock)
    columns : open, high, low, close, volume, turnover   (turnover = traded value in INR)

Providers
    yahoo   - yfinance, `<SYMBOL>.NS`.  Split/bonus/dividend ADJUSTED (auto_adjust=True), so the
              moving averages do not see fake gaps.  Daily history is long; intraday (60m) is
              limited by Yahoo to the last ~730 days.  Turnover is approximated as close * volume.
    nse     - official NSE "full bhavcopy" (sec_bhavdata_full_DDMMYYYY.csv), one file per day for
              every EQ-series stock, with exact traded value.  UNADJUSTED for corporate actions.
    csv     - a folder of `<SYMBOL>.csv` files (date,open,high,low,close,volume[,turnover]).

NSE and Yahoo blocked the sandbox this code was written in, so the network paths of `yahoo` and
`nse` have been exercised only against mocked responses; run `python -m indian_stocks fetch` once on
a machine with internet access to confirm them.
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .universe import NSE_HEADERS

OHLCV = ["open", "high", "low", "close", "volume", "turnover"]
IST_OPEN = (9, 15)
IST_CLOSE = (15, 30)


def ist_now() -> datetime:
    """Current IST wall-clock time as a naive datetime."""
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None)


# ───────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ───────────────────────────────────────────────────────────────────────────

def timeframe_hours(timeframe: str) -> Optional[int]:
    """'1d' -> None (daily).  '60m' -> 1, '1h' -> 1, '2h' -> 2, '4h' -> 4."""
    tf = timeframe.strip().lower()
    if tf in ("1d", "d", "24h", "day", "daily"):
        return None
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf in ("60m",):
        return 1
    raise ValueError(f"Unsupported timeframe '{timeframe}' (use 1d, 1h, 2h, 4h ...)")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, numeric dtype, tz-naive sorted index, drop empty rows, add turnover."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"missing '{c}' column (have {list(out.columns)})")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["high"] >= out["low"])]
    out["volume"] = out["volume"].fillna(0.0)
    if "turnover" not in out.columns:
        out["turnover"] = out["close"] * out["volume"]
    else:
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(out["close"] * out["volume"])
    return out[OHLCV]


def resample_session_bars(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Merge hourly NSE bars into N-hour bars anchored to the 09:15 open of each session.

    NSE has a 6h15m session, so 4H candles are 09:15-13:15 and 13:15-15:30 (a short last bar).
    Bars never straddle two sessions - unlike the original's 24/7 crypto candles.
    """
    if hours <= 1 or df.empty:
        return df
    d = df.copy()
    day = d.index.normalize()
    pos = d.groupby(day).cumcount() // hours
    key = pd.MultiIndex.from_arrays([day, pos])
    agg = d.groupby(key).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("volume", "sum"),
                             turnover=("turnover", "sum"))
    first_ts = d.groupby(key).apply(lambda g: g.index[0])
    agg.index = pd.DatetimeIndex(first_ts.values)
    return agg[OHLCV]


def drop_incomplete_last_bar(df: pd.DataFrame, timeframe: str = "1d",
                             now: Optional[datetime] = None) -> pd.DataFrame:
    """Drop the last bar if it is still forming (NSE closes 15:30 IST).

    The original always excluded the forming candle; strategy decisions must use completed bars.
    `now` is IST wall-clock (naive); defaults to the current IST time.
    """
    if df.empty:
        return df
    if now is None:
        now = ist_now()
    last = df.index[-1]
    hours = timeframe_hours(timeframe)
    if hours is None:
        market_closed = (now.hour, now.minute) >= IST_CLOSE
        if last.normalize() == pd.Timestamp(now.date()) and not market_closed:
            return df.iloc[:-1]
        if last.normalize() > pd.Timestamp(now.date()):
            return df.iloc[:-1]
    else:
        # intraday bars are stamped with their start time
        session_close = datetime.combine(last.date(), datetime.min.time()).replace(hour=IST_CLOSE[0], minute=IST_CLOSE[1])
        end = min(last.to_pydatetime() + timedelta(hours=hours), session_close)
        if end > now:
            return df.iloc[:-1]
    return df


# ───────────────────────────────────────────────────────────────────────────
#  CSV provider (offline / tests / your own data)
# ───────────────────────────────────────────────────────────────────────────

class CsvProvider:
    name = "csv"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def fetch(self, symbols: Sequence[str], start=None, end=None, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            path = self.data_dir / f"{sym}.csv"
            if not path.exists():
                continue
            raw = pd.read_csv(path)
            cols = {c: str(c).strip().lower() for c in raw.columns}
            raw = raw.rename(columns=cols)
            dcol = next((c for c in ("date", "datetime", "timestamp") if c in raw.columns), raw.columns[0])
            raw.index = pd.to_datetime(raw.pop(dcol))
            df = normalize_ohlcv(raw)
            out[sym] = _slice(df, start, end)
        return {s: d for s, d in out.items() if len(d)}


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df


# ───────────────────────────────────────────────────────────────────────────
#  Yahoo provider
# ───────────────────────────────────────────────────────────────────────────

def yahoo_ticker(symbol: str) -> str:
    """RELIANCE -> RELIANCE.NS ; indices (^NSEI) and already-suffixed tickers are left alone."""
    return symbol if symbol.startswith("^") or "." in symbol else f"{symbol}.NS"


class YahooProvider:
    name = "yahoo"

    def __init__(self, cache_dir: Optional[str] = "data/cache", cache_ttl_hours: float = 12.0,
                 batch_size: int = 40):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_hours * 3600
        self.batch_size = batch_size

    # -- cache ---------------------------------------------------------
    def _cache_path(self, sym: str, timeframe: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = sym.replace("&", "_AND_").replace("/", "_").replace("^", "IDX_")
        return self.cache_dir / "yahoo" / timeframe / f"{safe}.csv"

    def _read_cache(self, sym, timeframe, start, end) -> Optional[pd.DataFrame]:
        p = self._cache_path(sym, timeframe)
        if not p or not p.exists():
            return None
        df = normalize_ohlcv(pd.read_csv(p, index_col=0, parse_dates=True))
        if df.empty:
            return None
        recent_request = pd.Timestamp(end) >= pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
        stale = recent_request and (time.time() - p.stat().st_mtime) > self.cache_ttl
        covers_start = df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
        covers_end = df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5)
        return None if (stale or not covers_start or not covers_end) else df

    def _write_cache(self, sym, timeframe, df):
        p = self._cache_path(sym, timeframe)
        if p and len(df):
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p)

    # -- download ------------------------------------------------------
    def _download(self, tickers: List[str], start, end, interval: str) -> pd.DataFrame:
        import yfinance as yf
        return yf.download(tickers, start=str(pd.Timestamp(start).date()),
                           end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                           interval=interval, auto_adjust=True, group_by="ticker",
                           progress=False, threads=True)

    @staticmethod
    def _extract(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=OHLCV)
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            lvl1 = raw.columns.get_level_values(1)
            if ticker in set(lvl0):
                return raw[ticker]
            if single and ticker in set(lvl1):
                return raw.xs(ticker, axis=1, level=1)
            return pd.DataFrame(columns=OHLCV)
        return raw if single else pd.DataFrame(columns=OHLCV)

    def fetch(self, symbols: Sequence[str], start, end, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        hours = timeframe_hours(timeframe)
        interval = "1d" if hours is None else "60m"
        out: Dict[str, pd.DataFrame] = {}
        todo: List[str] = []
        for s in symbols:
            cached = self._read_cache(s, timeframe, start, end)
            if cached is not None:
                out[s] = cached
            else:
                todo.append(s)

        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            tickers = [yahoo_ticker(s) for s in batch]
            try:
                raw = self._download(tickers, start, end, interval)
            except Exception as exc:
                print(f"   ! yahoo batch {i // self.batch_size} failed: {exc}", file=sys.stderr)
                continue
            for s, t in zip(batch, tickers):
                try:
                    df = normalize_ohlcv(self._extract(raw, t, single=len(batch) == 1))
                except Exception as exc:
                    print(f"   ! {s}: unusable Yahoo data ({exc})", file=sys.stderr)
                    continue
                if hours and hours > 1:
                    df = resample_session_bars(df, hours)
                if len(df):
                    self._write_cache(s, timeframe, df)
                    out[s] = df
        return {s: _slice(d, start, end) for s, d in out.items() if len(_slice(d, start, end))}


# ───────────────────────────────────────────────────────────────────────────
#  NSE bhavcopy provider (daily, official, unadjusted)
# ───────────────────────────────────────────────────────────────────────────

BHAVCOPY_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"


def parse_bhavcopy(text: str) -> pd.DataFrame:
    """Parse one day of NSE full bhavcopy into [symbol, date, open, high, low, close, volume, turnover].

    Only EQ series.  TURNOVER_LACS is in lakh INR; converted to INR.
    """
    raw = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    need = ["SYMBOL", "SERIES", "DATE1", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"bhavcopy is missing columns {missing} (have {list(raw.columns)})")
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].astype(str).str.strip()
    raw = raw[raw["SERIES"] == "EQ"].copy()
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    out = pd.DataFrame({
        "symbol": raw["SYMBOL"].str.upper(),
        "date": pd.to_datetime(raw["DATE1"], format="%d-%b-%Y", errors="coerce"),
        "open": num("OPEN_PRICE"), "high": num("HIGH_PRICE"), "low": num("LOW_PRICE"),
        "close": num("CLOSE_PRICE"), "volume": num("TTL_TRD_QNTY"),
    })
    if "TURNOVER_LACS" in raw.columns:
        out["turnover"] = num("TURNOVER_LACS") * 1e5
    else:
        out["turnover"] = out["close"] * out["volume"]
    return out.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)


class NSEBhavcopyProvider:
    name = "nse"

    def __init__(self, cache_dir: Optional[str] = "data/cache", session=None, pause_sec: float = 0.4):
        self.cache_dir = Path(cache_dir) / "bhavcopy" if cache_dir else None
        self.session = session
        self.pause = pause_sec
        self._warmed = False

    def _sess(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        if not self._warmed:
            try:  # NSE sets cookies on the home page; archives usually work without but be polite
                self.session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=20)
            except Exception:
                pass
            self._warmed = True
        return self.session

    def _day(self, d: date) -> Optional[pd.DataFrame]:
        if d.weekday() >= 5:
            return None
        cache = self.cache_dir / f"{d:%Y%m%d}.csv" if self.cache_dir else None
        miss = self.cache_dir / f"{d:%Y%m%d}.holiday" if self.cache_dir else None
        if cache and cache.exists():
            return parse_bhavcopy(cache.read_text(encoding="utf-8"))
        if miss and miss.exists():
            return None
        url = BHAVCOPY_URL.format(ddmmyyyy=f"{d:%d%m%Y}")
        try:
            resp = self._sess().get(url, headers=NSE_HEADERS, timeout=30)
        except Exception as exc:
            print(f"   ! {d}: download failed ({exc})", file=sys.stderr)
            return None
        time.sleep(self.pause)
        if resp.status_code == 404:           # market holiday
            if miss:
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.write_text("", encoding="utf-8")
            return None
        if resp.status_code != 200:
            print(f"   ! {d}: HTTP {resp.status_code}", file=sys.stderr)
            return None
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(resp.text, encoding="utf-8")
        return parse_bhavcopy(resp.text)

    def fetch_days(self, start, end) -> pd.DataFrame:
        frames = []
        d = pd.Timestamp(start).date()
        last = pd.Timestamp(end).date()
        while d <= last:
            f = self._day(d)
            if f is not None and len(f):
                frames.append(f)
            d += timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch(self, symbols: Optional[Sequence[str]], start, end, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        if timeframe_hours(timeframe) is not None:
            raise ValueError("The NSE bhavcopy provider is daily only; use --provider yahoo for intraday")
        allrows = self.fetch_days(start, end)
        if allrows.empty:
            return {}
        wanted = None if symbols is None else {s.upper() for s in symbols}
        out: Dict[str, pd.DataFrame] = {}
        for sym, g in allrows.groupby("symbol"):
            if wanted is not None and sym not in wanted:
                continue
            df = g.set_index("date")[OHLCV]
            out[sym] = normalize_ohlcv(df)
        return out

    def turnover_snapshot(self, on: Optional[date] = None, lookback_days: int = 10) -> Dict[str, float]:
        """{symbol: traded value in INR} for the latest trading day on or before `on`."""
        d = on or ist_now().date()
        for _ in range(lookback_days):
            f = self._day(d)
            if f is not None and len(f):
                return dict(zip(f["symbol"], f["turnover"]))
            d -= timedelta(days=1)
        return {}


def get_provider(name: str, cache_dir: Optional[str] = "data/cache", data_dir: Optional[str] = None):
    name = name.lower()
    if name == "yahoo":
        return YahooProvider(cache_dir=cache_dir)
    if name == "nse":
        return NSEBhavcopyProvider(cache_dir=cache_dir)
    if name == "csv":
        if not data_dir:
            raise ValueError("--data-dir is required with --provider csv")
        return CsvProvider(data_dir)
    raise ValueError(f"Unknown provider '{name}' (yahoo | nse | csv)")
