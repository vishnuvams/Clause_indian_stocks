"""Candle frames with precomputed EMAs and a "last completed candle" lookup.

Timestamps are epoch-milliseconds of the tz-naive IST wall clock (only differences and ordering
matter).  Every bar has an open time and a close time:

    daily bar   : 09:15 -> 15:30 of its date
    intraday bar: start -> min(start + N hours, 15:30)

so "completed at time t" means `t_close <= t`, which makes a daily frame and an hourly frame line up
correctly inside one NSE session (the original ran 24/7 and used UTC-aligned candles).

EMA recursion runs over the whole history; the live bot recomputes it over its last 250 candles.
The two agree to well under 0.1% once ~100 candles of warm-up exist (seed influence decays
geometrically), so decisions are unaffected.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

MIN_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
OPEN_OFFSET_MS = (9 * 60 + 15) * MIN_MS      # 09:15
CLOSE_OFFSET_MS = (15 * 60 + 30) * MIN_MS    # 15:30


def _to_ms(idx: pd.DatetimeIndex) -> List[int]:
    # pandas >= 3 defaults to microsecond resolution (older: nanoseconds) - be explicit.
    return [int(v) for v in idx.as_unit("ms").asi8]


def ema_series(closes: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return out
    k = 2.0 / (period + 1)
    e = sum(closes[:period]) / period
    out[period - 1] = e
    for i in range(period, len(closes)):
        e = closes[i] * k + e * (1 - k)
        out[i] = e
    return out


class Frame:
    """OHLCV bars at one timeframe + EMA series."""

    def __init__(self, t_open: List[int], t_close: List[int], o, h, l, c, turnover,
                 periods: Iterable[int] = (9, 21, 50)):
        self.t_open, self.t_close = t_open, t_close
        self.o, self.h, self.l, self.c, self.turnover = o, h, l, c, turnover
        self._ema: Dict[int, List[Optional[float]]] = {p: ema_series(c, p) for p in periods}
        self._psum = [0.0]
        for x in c:
            self._psum.append(self._psum[-1] + x)

    def __len__(self):
        return len(self.c)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_df(cls, df: pd.DataFrame, hours: Optional[int], periods=(9, 21, 50)) -> "Frame":
        """`hours=None` -> daily bars stamped at midnight; otherwise intraday bars stamped at start."""
        idx = df.index
        if hours is None:
            day0 = _to_ms(idx.normalize())
            t_open = [d + OPEN_OFFSET_MS for d in day0]
            t_close = [d + CLOSE_OFFSET_MS for d in day0]
        else:
            t_open = _to_ms(idx)
            day0 = _to_ms(idx.normalize())
            t_close = [min(o + hours * HOUR_MS, d + CLOSE_OFFSET_MS) for o, d in zip(t_open, day0)]
        return cls(t_open, t_close,
                   df["open"].astype(float).tolist(), df["high"].astype(float).tolist(),
                   df["low"].astype(float).tolist(), df["close"].astype(float).tolist(),
                   df["turnover"].astype(float).tolist(), periods)

    # -- lookups --------------------------------------------------------
    def done_index(self, ts: int) -> int:
        """Index of the last bar fully closed at time `ts` (-1 if none)."""
        return bisect_right(self.t_close, ts) - 1

    def ema_done(self, period: int, k: int) -> Optional[float]:
        """EMA over closes[0..k]  (== calculate_ema(done_closes))."""
        if k < 0:
            return None
        if period not in self._ema:
            self._ema[period] = ema_series(self.c, period)
        return self._ema[period][k]

    def ema_live(self, period: int, k: int, x: float) -> Optional[float]:
        """EMA over closes[0..k] + [x]  (== calculate_ema(done + in-progress close x))."""
        n = k + 1
        if n + 1 < period:
            return None
        if n + 1 == period:
            return (self._psum[n] + x) / period
        prev = self.ema_done(period, k)
        if prev is None:
            return None
        kk = 2.0 / (period + 1)
        return x * kk + prev * (1 - kk)

    def closes_done(self, k: int, n: Optional[int] = None) -> List[float]:
        lo = 0 if n is None else max(0, k + 1 - n)
        return self.c[lo:k + 1]

    def triple_done(self, k: int, periods=(9, 21, 50)) -> Tuple[Optional[float], ...]:
        return tuple(self.ema_done(p, k) for p in periods)

    def triple_live(self, k: int, x: float, periods=(9, 21, 50)) -> Tuple[Optional[float], ...]:
        return tuple(self.ema_live(p, k, x) for p in periods)
