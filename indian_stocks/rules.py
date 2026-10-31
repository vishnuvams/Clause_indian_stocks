"""Pure strategy rules - ported 1:1 from `ema_9_21_50_stocks_crypto_1day_1hour.py`.

Function names mirror the live bot's methods so each can be diffed against the original:

    ema_value                  EMAPullbackTrader.calculate_ema
    weak_run_decision          _weak_run_decision          (the "lower-close" exit)
    check_exit                 EMAPullbackTrader._check_exit
    raw_setup                  EMAPullbackTrader._raw_setup
    closed_candle_confirms     _closed_candle_confirms
    daily_ribbon_confirms      _daily_ribbon_confirms      (gate enabled)
    would_fill_immediately     _would_fill_immediately
    near_extreme_dist/stats    _near_extreme_dist / _near_extreme_stats
    limit_extension_pct        _limit_extension_pct
    lower_high_stats/fails     _lower_high_stats / _lower_high_fails
    ribbon_ok                  EMAPullbackBot._ribbon_ok   (selection stage)
    adr_pct                    _compute_adr_pct / _adr_pct_from_daily
    nday_return_frac           _nday_return_frac
    add_size_pct               _add_size_pct
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

LONG = "LONG"
SHORT = "SHORT"
BANDS = ("EMA9", "EMA21", "EMA50")
_BAND = {"ema9": "EMA9", "ema21": "EMA21", "ema50": "EMA50"}


def band_label(name, default="EMA9"):
    return _BAND.get(str(name).strip().lower(), default)


# ───────────────────────────── indicators ─────────────────────────────

def ema_value(closes: Sequence[float], period: int) -> Optional[float]:
    """Exponential MA. Seed = SMA of the first `period` closes, then EMA recursion."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def adr_pct(highs: Sequence[float], lows: Sequence[float]) -> Optional[float]:
    """Average daily range % = (mean(high/low) - 1) * 100 over the supplied COMPLETED candles.
    None if fewer than 3 usable candles."""
    ratios = [h / l for h, l in zip(highs, lows) if l > 0 and h > 0]
    if len(ratios) < 3:
        return None
    return (sum(ratios) / len(ratios) - 1.0) * 100.0


def nday_return_frac(closes_done: Sequence[float], n: int) -> Optional[float]:
    """close[-1] / close[-1-n] - 1 over COMPLETED closes; None if history is too short."""
    if n <= 0 or len(closes_done) < n + 1:
        return None
    then = closes_done[-1 - n]
    if then <= 0:
        return None
    return closes_done[-1] / then - 1.0


# ───────────────────────────── exits ─────────────────────────────

def weak_run_decision(closes, period, is_long, max_closes):
    """Lower-close exit decision from COMPLETED closes (oldest -> newest).

    A "weak run" = the consecutive latest closes on the wrong side of the `period` EMA
    (LONG: below, SHORT: above).  Inside a run, exit on the first close WORSE than the one
    before it (LONG: lower, SHORT: higher); a close back across the EMA ends the run (no exit).
    A run of `max_closes` closes with no worse close also exits (time limit).
    Returns (exit, run_length, kind, close_now, close_prev); kind in {"lower_close","time_limit",""}.
    """
    n = len(closes)
    if n < period + 1:
        return False, 0, "", None, None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    series = [None] * (period - 1) + [ema]
    for p in closes[period:]:
        ema = p * k + ema * (1 - k)
        series.append(ema)
    run, i = 0, n - 1
    while i >= period - 1 and ((closes[i] < series[i]) if is_long else (closes[i] > series[i])):
        run += 1
        i -= 1
    if run == 0:
        return False, 0, "", None, None
    start = n - run
    for j in range(start + 1, n):
        if (closes[j] < closes[j - 1]) if is_long else (closes[j] > closes[j - 1]):
            return True, run, "lower_close", closes[j], closes[j - 1]
    if run >= max_closes:
        return True, run, "time_limit", closes[-1], closes[-2] if n > 1 else None
    return False, run, "", None, None


def check_exit(is_long: bool, closes_done: Sequence[float], e9, e21, e50,
               exit_mode: str, exit_profit_2closes: str, max_closes: int,
               ema_fast=9, ema_mid=21, ema_slow=50) -> Optional[str]:
    """`_check_exit` on completed-candle ("done") EMAs.  Returns the exit reason or None.

    1) lower-close rule on the reference EMA (exit_profit_2closes), in profit AND loss;
    2) ribbon flip per exit_mode.
    """
    side = "below" if is_long else "above"
    period = {"ema50": ema_slow, "ema21": ema_mid, "ema9": ema_fast}.get(str(exit_profit_2closes).lower())
    if period and closes_done:
        trig, run, kind, c_now, c_prev = weak_run_decision(closes_done, period, is_long, max_closes)
        if trig:
            ref = str(exit_profit_2closes).upper()
            if kind == "lower_close":
                what = "lower" if is_long else "higher"
                return f"{what} close while {side} {ref} ({c_now:.2f} vs previous {c_prev:.2f}, {run} closes {side} {ref})"
            return f"{run} closes {side} {ref} without recovering (time limit {max_closes})"

    mode = str(exit_mode).strip().lower()
    if mode == "ema9_ema50":
        if e9 is None or e50 is None:
            return None
        flipped = (e9 <= e50) if is_long else (e9 >= e50)
    elif mode == "ema9_ema21":
        if e9 is None or e21 is None:
            return None
        flipped = (e9 <= e21) if is_long else (e9 >= e21)
    else:  # "ema9_ema21_ema50" - full ribbon
        if e9 is None or e21 is None or e50 is None:
            return None
        flipped = (e9 <= e21 <= e50) if is_long else (e9 >= e21 >= e50)
    return f"{mode} flip" if flipped else None


# ───────────────────────────── entries ─────────────────────────────

def raw_setup(e9, e21, e50, enable_shorts: bool,
              band_for: Callable[[str], str]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """(direction, band, limit_price) from the LIVE ribbon alone, else (None, None, None).
    The full 3-EMA ribbon is required."""
    if e9 is None or e21 is None or e50 is None:
        return None, None, None
    if e9 >= e21 >= e50:
        direction = LONG
    elif enable_shorts and e9 <= e21 <= e50:
        direction = SHORT
    else:
        return None, None, None
    band = band_for(direction)
    return direction, band, {"EMA9": e9, "EMA21": e21, "EMA50": e50}[band]


def closed_candle_confirms(direction, closed_close, e9, e21, e50, band) -> bool:
    """The LAST COMPLETED candle must have closed beyond the BAND EMA and every SLOWER EMA."""
    vals = {"EMA9": e9, "EMA21": e21, "EMA50": e50}
    idx = BANDS.index(band) if band in BANDS else 0
    refs = [vals[k] for k in BANDS[idx:]]
    if any(r is None for r in refs) or not closed_close or closed_close <= 0:
        return False
    return all(closed_close > r for r in refs) if direction == LONG else all(closed_close < r for r in refs)


def would_fill_immediately(direction, limit_price, live_price) -> bool:
    """A resting pullback limit must sit BELOW the market for a LONG (ABOVE for a SHORT)."""
    if not live_price or live_price <= 0:
        return False
    return live_price <= limit_price if direction == LONG else live_price >= limit_price


def daily_ribbon_confirms(direction, e9, e21, e50) -> bool:
    """Daily EMA9>EMA21>EMA50 (mirror for shorts); falls back to EMA9 vs EMA21 if EMA50 is not
    available yet; fails if EMA21 is unavailable."""
    if e9 is None or e21 is None:
        return False
    if e50 is not None:
        return (e9 > e21 > e50) if direction == LONG else (e9 < e21 < e50)
    return (e9 > e21) if direction == LONG else (e9 < e21)


def ribbon_ok(e9, e21, e50) -> bool:
    """Selection-stage pre-filter: the DAILY ribbon is monotonically stacked in EITHER direction."""
    if e9 is None or e21 is None:
        return False
    if e50 is not None:
        return (e9 > e21 > e50) or (e9 < e21 < e50)
    return e9 != e21


def near_extreme_stats(done_closes: Sequence[float], lookback: int):
    """(last_close, highest_close, lowest_close, n) over the last `lookback` COMPLETED closes."""
    done = list(done_closes)[-lookback:]
    if len(done) < 2:
        return None
    return done[-1], max(done), min(done), len(done)


def near_extreme_dist(direction, last, hi, lo) -> float:
    if direction == LONG:
        return (hi - last) / hi if hi > 0 else 0.0
    return (last - lo) / lo if lo > 0 else 0.0


def limit_extension_pct(direction, hi, lo, limit_price) -> float:
    if limit_price <= 0:
        return 0.0
    if direction == LONG:
        return hi / limit_price - 1.0
    return (limit_price - lo) / limit_price


def lower_high_stats(closes_done: Sequence[float], recent_days: int, prior_days: int):
    """(recent_high, prior_high, recent_low, prior_low); None if not enough history."""
    if len(closes_done) < recent_days + prior_days:
        return None
    recent = closes_done[-recent_days:]
    prior = closes_done[-(recent_days + prior_days):-recent_days]
    return max(recent), max(prior), min(recent), min(prior)


def lower_high_fails(direction, stats) -> bool:
    rh, ph, rl, pl = stats
    return (rh < ph) if direction == LONG else (rl > pl)


# ───────────────────────────── sizing ─────────────────────────────

def add_size_pct(base: float, step: float, floor: float, adds_done: int) -> float:
    """base - step * adds_done, floored: 50, 40, 30, 20, 10, 10 ..."""
    pct = base - max(0.0, step) * adds_done
    return max(pct, min(max(0.0, floor), base))
