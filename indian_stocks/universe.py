"""Stock LIST handling - which NSE symbols are even considered.

The strategy's selection FILTERS (volume floor, ADR, relative strength, ribbon, near-high, ...) are
in `selection.py`.  This module only builds the candidate list, the NSE analogue of the original's
"every asset on the dex" + `_is_equity` (which dropped commodities / forex):

    NSE Nifty 50/100/200/500 or every EQ-series symbol   (ordinary company shares only:
    ETFs, index funds, InvITs, REITs, T2T/SME series are dropped by `is_stock`)
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

NSE_LIST_URLS: Dict[str, str] = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "all": "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
}

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

FALLBACK_CSV = Path(__file__).resolve().parent.parent / "data" / "universe_fallback.csv"

# Name / symbol patterns for things that trade in the EQ series but are not operating companies.
_NON_STOCK_NAME = re.compile(
    r"\b(ETF|EXCHANGE TRADED|INDEX FUND|MUTUAL FUND|LIQUID|GILT|INVIT|REIT|"
    r"INFRASTRUCTURE INVESTMENT TRUST|REAL ESTATE INVESTMENT TRUST|NIFTY|SENSEX|BEES)\b",
    re.IGNORECASE,
)
_NON_STOCK_SYMBOL = re.compile(r"(BEES|ETF|IETF)$|^(NIFTY|BANKNIFTY|SENSEX|LIQUID)", re.IGNORECASE)
_BAD_SYMBOL_SUFFIX = re.compile(r"-(RE\d*|W\d*|BE|BZ|SM|ST|N\d|Y\d|GB)$", re.IGNORECASE)
_VALID_SYMBOL = re.compile(r"^[A-Z0-9&\-]+$")


def is_stock(symbol: str, name: str = "", series: str = "EQ") -> bool:
    """True for an ordinary NSE-listed company share (the analogue of `_is_equity`).

    Heuristic - it can miss a newly launched fund.  Use `blacklist` for strays.
    """
    s = (symbol or "").strip().upper()
    if not s or not _VALID_SYMBOL.match(s):
        return False
    if series and series.strip().upper() != "EQ":
        return False            # BE = trade-to-trade, SM/ST = SME, GB = gold bonds, ...
    if _BAD_SYMBOL_SUFFIX.search(s) or _NON_STOCK_SYMBOL.search(s):
        return False
    if name and _NON_STOCK_NAME.search(name):
        return False
    return True


def parse_universe_csv(text: str) -> pd.DataFrame:
    """Parse an NSE index-constituent CSV or EQUITY_L.csv into [symbol, name, series]."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    sym_col = next((c for c in df.columns if c == "symbol"), None)
    name_col = next((c for c in df.columns if c in ("company name", "name of company", "name")), None)
    ser_col = next((c for c in df.columns if c == "series"), None)
    if sym_col is None:
        raise ValueError(f"No SYMBOL column in universe CSV (columns: {list(df.columns)})")
    out = pd.DataFrame({
        "symbol": df[sym_col].astype(str).str.strip().str.upper(),
        "name": df[name_col].astype(str).str.strip() if name_col else "",
        "series": df[ser_col].astype(str).str.strip().str.upper() if ser_col else "EQ",
    })
    return out.drop_duplicates("symbol").reset_index(drop=True)


def filter_stocks(df: pd.DataFrame, blacklist: Iterable[str] = ()) -> pd.DataFrame:
    bl = {b.strip().upper() for b in blacklist}
    mask = [
        is_stock(r.symbol, r.name, r.series) and r.symbol not in bl
        for r in df.itertuples(index=False)
    ]
    return df[mask].reset_index(drop=True)


def load_universe(name: str = "nifty500", cache_dir: Optional[str] = "data/cache",
                  blacklist: Iterable[str] = (), session=None, offline: bool = False) -> pd.DataFrame:
    """Return the stock list [symbol, name, series] for `name`.

    `name`: nifty50 | nifty100 | nifty200 | nifty500 | all | file:<path to csv>.
    Order of attempts: local cache -> download from NSE archives -> bundled fallback list.
    """
    name = name.strip()
    if name.lower().startswith("file:"):
        path = name[5:]
        text = Path(path).read_text(encoding="utf-8")
        return filter_stocks(parse_universe_csv(text), blacklist)

    key = name.lower()
    if key not in NSE_LIST_URLS:
        raise ValueError(f"Unknown universe '{name}'. Choose from {sorted(NSE_LIST_URLS)} or file:<path>")

    cache_path = Path(cache_dir) / f"universe_{key}.csv" if cache_dir else None
    text: Optional[str] = None

    if not offline:
        try:
            import requests
            sess = session or requests.Session()
            resp = sess.get(NSE_LIST_URLS[key], headers=NSE_HEADERS, timeout=30)
            resp.raise_for_status()
            text = resp.text
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
        except Exception as exc:  # network blocked, NSE rate-limit, format change ...
            print(f"   ! could not download NSE list '{key}': {exc}", file=sys.stderr)

    if text is None and cache_path and cache_path.exists():
        print(f"   i using cached NSE list {cache_path}", file=sys.stderr)
        text = cache_path.read_text(encoding="utf-8")

    if text is None:
        print(f"   ! falling back to the bundled starter list {FALLBACK_CSV} - "
              f"it is a short static list of large caps, NOT the real '{key}' membership.",
              file=sys.stderr)
        text = FALLBACK_CSV.read_text(encoding="utf-8")

    return filter_stocks(parse_universe_csv(text), blacklist)


def select_by_turnover(turnover: Mapping[str, float], min_turnover_inr: float,
                       blacklist: Iterable[str] = ()) -> List[Tuple[str, float]]:
    """Keep symbols with traded value >= min_turnover_inr, sorted highest first."""
    bl = {b.strip().upper() for b in blacklist}
    rows = [(s, float(v)) for s, v in turnover.items()
            if s not in bl and v is not None and float(v) >= min_turnover_inr]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
