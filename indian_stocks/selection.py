"""Stock selection - port of `EMAPullbackBot._refresh_stocks` (+ `_update_market_regime`).

A symbol makes the watchlist only if it clears, in this order:

    1. volume floor      traded value of the last completed day >= min_turnover_inr
    2. ADR floor         mean(high/low)-1 over the last 14 completed daily candles >= 4%
    3. relative strength 30d return minus the benchmark's >= +10 pts  -> LONG leader
                         (<= -10 pts and shorts enabled              -> SHORT laggard)
                         RS both selects the stock AND locks its side for every later entry.
    4. daily ribbon      LIVE daily EMA9/21/50 monotonically stacked (either way)
    5. near-the-high     last completed close within 10% of its 30-day closing high (SHORT: low)
    6. recent trend      A: 10d return beats the benchmark's 10d return (SHORT: trails it)
                         B: EMA21 up >= 1% over 5 days (SHORT: down)
    7. lower-high        highest close of days 1-5 ago must not be below that of days 6-10 ago
                         (SHORT: lowest close must not be above)

Missing history passes the test that needs it (the other gates decide) exactly as in the original.
Everything uses COMPLETED daily candles except the daily EMAs of step 4, which include the
in-progress candle at its current price (here: the price at the cycle time).

`select_symbol` returns a `Selection`; when it is rejected, `.reason` says which gate dropped it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import rules
from .config import StrategyConfig
from .frames import Frame


@dataclass
class Selection:
    ok: bool
    direction: Optional[str] = None     # RS side (LONG / SHORT) or None when RS is off
    rs: Optional[float] = None          # relative strength, fraction (0.10 = +10 pts)
    adr: Optional[float] = None         # %
    turnover: float = 0.0
    reason: str = ""
    e9: Optional[float] = None
    e21: Optional[float] = None
    e50: Optional[float] = None


def _fail(reason: str, **kw) -> Selection:
    return Selection(False, reason=reason, **kw)


def select_symbol(D: Frame, kd: int, x: float, B: Optional[Frame], kb: int,
                  cfg: StrategyConfig) -> Selection:
    """Evaluate one symbol at a point in time.

    D  : the symbol's DAILY frame           kd : index of its last completed daily bar
    x  : live price (the in-progress daily candle's close)
    B  : benchmark daily frame (or None)    kb : index of the benchmark's last completed bar
    """
    if kd < 2:
        return _fail("not enough daily history")
    window = cfg.window_candles - 1                 # 249 completed candles, as the live fetch
    lo = max(0, kd + 1 - window)
    turnover = D.turnover[kd]

    # 1. volume floor
    if cfg.min_turnover_inr > 0 and turnover < cfg.min_turnover_inr:
        return _fail(f"traded value {turnover / 1e7:.2f} cr below floor {cfg.min_turnover_inr / 1e7:.2f} cr",
                     turnover=turnover)

    # 2. ADR floor
    adr = None
    if cfg.adr_min_pct > 0:
        a_lo = max(lo, kd + 1 - cfg.adr_period)
        adr = rules.adr_pct(D.h[a_lo:kd + 1], D.l[a_lo:kd + 1])
        if adr is None:
            return _fail("no ADR data", turnover=turnover)
        if adr < cfg.adr_min_pct:
            return _fail(f"ADR {adr:.2f}% below {cfg.adr_min_pct:.1f}%", adr=adr, turnover=turnover)

    done = D.c[lo:kd + 1]

    # 3. relative strength -> side
    direction = None
    rs = None
    if cfg.rs_enabled:
        raw = rules.nday_return_frac(done, cfg.rs_lookback_days)
        if raw is None:
            return _fail("not enough history for RS", adr=adr, turnover=turnover)
        bench = None
        if B is not None and kb >= 0:
            bench = rules.nday_return_frac(B.c[max(0, kb + 1 - window):kb + 1], cfg.rs_lookback_days)
        rs = raw - (bench if bench is not None else 0.0)
        if rs >= cfg.rs_min_pct_long / 100.0:
            direction = rules.LONG
        elif cfg.enable_shorts and rs <= -cfg.rs_min_pct_short / 100.0:
            direction = rules.SHORT
        else:
            return _fail(f"RS {rs * 100:+.1f} pts not beyond +{cfg.rs_min_pct_long:.0f}"
                         + (f" / -{cfg.rs_min_pct_short:.0f}" if cfg.enable_shorts else ""),
                         rs=rs, adr=adr, turnover=turnover)

    base = dict(direction=direction, rs=rs, adr=adr, turnover=turnover)

    # 4. daily ribbon (live EMAs)
    e9, e21, e50 = D.triple_live(kd, x, (cfg.ema_fast, cfg.ema_mid, cfg.ema_slow))
    base.update(e9=e9, e21=e21, e50=e50)
    if not rules.ribbon_ok(e9, e21, e50):
        return _fail("daily EMA ribbon not stacked", **base)

    d = direction or (rules.LONG if (e9 is not None and e21 is not None and e9 >= e21) else rules.SHORT)
    is_long = d == rules.LONG

    # 5. near-the-high
    if cfg.near_high_max_pct > 0:
        st = rules.near_extreme_stats(done, cfg.near_high_lookback)
        if st is not None:
            dist = rules.near_extreme_dist(d, st[0], st[1], st[2])
            if dist > cfg.near_high_max_pct:
                return _fail(f"{dist * 100:.1f}% from {cfg.near_high_lookback}d closing "
                             f"{'high' if is_long else 'low'} (limit {cfg.near_high_max_pct * 100:.0f}%)", **base)

    # 6. recent trend: A (recent RS) + B (EMA21 slope)
    if cfg.recent_rs_filter or cfg.ema21_rising_filter:
        n, sd = cfg.recent_rs_days, cfg.ema21_slope_days
        ret_recent = rules.nday_return_frac(done, n)
        slope = None
        now_e = rules.ema_value(done, cfg.ema_mid)
        prev_e = rules.ema_value(done[:-sd], cfg.ema_mid)
        if now_e is not None and prev_e is not None and prev_e > 0:
            slope = (now_e / prev_e - 1.0) * 100.0
        if cfg.recent_rs_filter and ret_recent is not None and B is not None and kb >= 0:
            b_recent = rules.nday_return_frac(B.c[max(0, kb + 1 - window):kb + 1], n)
            if b_recent is not None:
                rel = ret_recent * 100.0 - b_recent * 100.0
                ok = (rel >= cfg.recent_rs_min_pct) if is_long else (rel <= -cfg.recent_rs_min_pct)
                if not ok:
                    return _fail(f"{n}d return {ret_recent * 100:+.1f}% vs benchmark {b_recent * 100:+.1f}%", **base)
        if cfg.ema21_rising_filter and slope is not None:
            ok = (slope >= cfg.ema21_slope_min_pct) if is_long else (slope <= -cfg.ema21_slope_min_pct)
            if not ok:
                return _fail(f"EMA21 {slope:+.1f}% over {sd}d", **base)

    # 7. lower-high
    if cfg.lower_high_filter:
        st = rules.lower_high_stats(done, cfg.lower_high_recent_days, cfg.lower_high_prior_days)
        if st is not None and rules.lower_high_fails(d, st):
            return _fail("making lower highs" if is_long else "making higher lows", **base)

    return Selection(True, direction, rs, adr, turnover, "selected", e9, e21, e50)


def regime_allowed(B: Optional[Frame], kb: int, cfg: StrategyConfig, previous: Optional[str]) -> Optional[str]:
    """`_update_market_regime`: benchmark close vs its EMA21 -> "LONG" / "SHORT" (None when the
    filter is off; keeps `previous` inside the buffer or when data is missing)."""
    if not cfg.regime_filter_enabled or B is None:
        return None
    if kb < cfg.regime_ema_period + 1:
        return previous
    done = B.c[max(0, kb + 1 - (cfg.window_candles - 1)):kb + 1]
    ema = rules.ema_value(done, cfg.regime_ema_period)
    if ema is None or ema <= 0:
        return previous
    close, buf = done[-1], cfg.regime_buffer_pct
    if close > ema * (1 + buf):
        return rules.LONG
    if close < ema * (1 - buf):
        return rules.SHORT
    return previous
