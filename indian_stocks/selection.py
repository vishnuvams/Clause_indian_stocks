"""Stock selection - port of `EMAPullbackBot._refresh_stocks` (+ `_update_market_regime`), stage for stage.
LONG-ONLY: the original's SHORT laggards are not part of this port.

`build_watchlist` takes the whole candidate set and returns the ordered watchlist exactly as the original
builds it with shorts disabled (same stages, same order of stages, same tie-breaking):

    1. volume floor       traded value of the last completed day >= min_turnover_inr
    2. ADR floor          mean(high/low)-1 over the last 14 completed daily candles >= 4 %    (no data => dropped)
    3. relative strength  rs = 30d return - benchmark's 30d return (raw return if no benchmark data)
                          ranked high -> low; leaders: rs >= +10 pts (top `rs_top_n_long` if > 0).
                          The cap is applied HERE, i.e. to the ADR survivors, BEFORE the filters below - so
                          fewer than N can survive.  RS off: every stock is a candidate, ordered by traded value.
    4. daily ribbon       live daily EMA9/21/50 monotonically stacked
    5. near-the-high      last completed close within 10 % of its 30d closing high
    6. recent trend       A: 10d return beats the benchmark's (skipped if benchmark data is missing)
                          B: EMA21 up >= 1 % over 5 days
    7. lower-high         highest close of days 1-5 ago not below that of days 6-10 ago

Missing history passes the test that needs it, exactly as in the original.  Everything uses COMPLETED daily
candles (the last 249, like the live bot's 250-candle fetch) except the daily EMAs of step 4, which include the
in-progress candle at its current price `x`.

`select_symbol` is the one-stock convenience wrapper; it also says WHICH gate dropped a stock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from . import rules
from .config import StrategyConfig
from .frames import Frame


@dataclass
class Selection:
    ok: bool
    direction: Optional[str] = None     # "LONG" for a selected stock
    rs: Optional[float] = None          # relative strength, fraction (0.10 = +10 pts)
    adr: Optional[float] = None         # %
    turnover: float = 0.0
    reason: str = ""
    e9: Optional[float] = None
    e21: Optional[float] = None
    e50: Optional[float] = None
    symbol: str = ""


@dataclass
class Candidate:
    """One stock at one point in time."""
    symbol: str
    D: Frame            # daily frame
    kd: int             # index of its last COMPLETED daily bar
    x: float            # live price (the in-progress daily candle's close)


def _window(cfg: StrategyConfig) -> int:
    return cfg.window_candles - 1


def _metrics(c: Candidate, cfg: StrategyConfig) -> Optional[dict]:
    D, kd = c.D, c.kd
    if kd < 2:
        return None
    lo = max(0, kd + 1 - _window(cfg))
    done = D.c[lo:kd + 1]
    a_lo = max(lo, kd + 1 - cfg.adr_period)
    e9, e21, e50 = D.triple_live(kd, c.x, (cfg.ema_fast, cfg.ema_mid, cfg.ema_slow))
    sd = cfg.ema21_slope_days
    now_e, prev_e = rules.ema_value(done, cfg.ema_mid), rules.ema_value(done[:-sd], cfg.ema_mid)
    slope = (now_e / prev_e - 1.0) * 100.0 if (now_e is not None and prev_e is not None and prev_e > 0) else None
    ret_recent = rules.nday_return_frac(done, cfg.recent_rs_days)
    return {
        "turnover": D.turnover[kd],
        "adr": rules.adr_pct(D.h[a_lo:kd + 1], D.l[a_lo:kd + 1]),
        "raw": rules.nday_return_frac(done, cfg.rs_lookback_days) if cfg.rs_enabled else None,
        "e9": e9, "e21": e21, "e50": e50, "ribbon_ok": rules.ribbon_ok(e9, e21, e50),
        "near": rules.near_extreme_stats(done, cfg.near_high_lookback),
        "ret_recent": None if ret_recent is None else ret_recent * 100.0,
        "slope": slope,
        "lh": rules.lower_high_stats(done, cfg.lower_high_recent_days, cfg.lower_high_prior_days),
    }


def _bench_returns(B: Optional[Frame], kb: int, cfg: StrategyConfig) -> Tuple[Optional[float], Optional[float]]:
    """(benchmark RS-window return as a fraction, benchmark recent-window return in PERCENT)."""
    if B is None or kb < 0:
        return None, None
    done = B.c[max(0, kb + 1 - _window(cfg)):kb + 1]
    rs = rules.nday_return_frac(done, cfg.rs_lookback_days)
    rec = rules.nday_return_frac(done, cfg.recent_rs_days)
    return rs, None if rec is None else rec * 100.0


def build_watchlist(cands: Iterable[Candidate], B: Optional[Frame], kb: int, cfg: StrategyConfig,
                    blacklist: Iterable[str] = ()) -> Tuple[List[Selection], Dict[str, str]]:
    """Run the whole selection pipeline.  Returns (ordered watchlist, {dropped symbol: reason})."""
    bl = {b.strip().upper() for b in blacklist}
    dropped: Dict[str, str] = {}
    rows: List[dict] = []
    for c in cands:
        if c.symbol in bl:
            dropped[c.symbol] = "blacklisted"
            continue
        m = _metrics(c, cfg)
        if m is None:
            dropped[c.symbol] = "not enough daily history"
            continue
        m["symbol"] = c.symbol
        rows.append(m)

    # 1. volume floor
    if cfg.min_turnover_inr > 0:
        keep = []
        for m in rows:
            if m["turnover"] >= cfg.min_turnover_inr:
                keep.append(m)
            else:
                dropped[m["symbol"]] = (f"traded value {m['turnover'] / 1e7:.2f} cr below floor "
                                        f"{cfg.min_turnover_inr / 1e7:.2f} cr")
        rows = keep

    # 2. ADR floor
    if cfg.adr_min_pct > 0:
        keep = []
        for m in rows:
            if m["adr"] is None:
                dropped[m["symbol"]] = "no ADR data"
            elif m["adr"] >= cfg.adr_min_pct:
                keep.append(m)
            else:
                dropped[m["symbol"]] = f"ADR {m['adr']:.2f}% below {cfg.adr_min_pct:.1f}%"
        rows = keep

    # 3. relative strength (+ ranking, cap)
    bench_rs, bench_recent = _bench_returns(B, kb, cfg)
    if cfg.rs_enabled and rows:
        for m in rows:
            m["rs"] = None if m["raw"] is None else (m["raw"] if bench_rs is None else m["raw"] - bench_rs)
            if m["rs"] is None:
                dropped[m["symbol"]] = "not enough history for RS"
        ranked = sorted((m for m in rows if m["rs"] is not None), key=lambda m: m["rs"], reverse=True)
        thr = cfg.rs_min_pct_long / 100.0
        long_ranked = [m for m in ranked if m["rs"] >= thr]
        leaders = long_ranked[:cfg.rs_top_n_long] if cfg.rs_top_n_long > 0 else long_ranked
        keep_syms = {m["symbol"] for m in leaders}
        for m in ranked:
            if m["symbol"] not in keep_syms:
                dropped[m["symbol"]] = (f"RS {m['rs'] * 100:+.1f} pts not beyond +{cfg.rs_min_pct_long:.0f}"
                                        if m["rs"] < thr else "cut by the rs_top_n cap")
        rows = [m for m in ranked if m["symbol"] in keep_syms]
    else:
        for m in rows:
            m["rs"] = None
        rows.sort(key=lambda m: m["turnover"], reverse=True)

    # 4. daily ribbon
    keep = []
    for m in rows:
        if m["ribbon_ok"]:
            keep.append(m)
        else:
            dropped[m["symbol"]] = "daily EMA ribbon not stacked"
    rows = keep

    # 5. near-the-high
    if cfg.near_high_max_pct > 0:
        keep = []
        for m in rows:
            st = m["near"]
            if st is None:
                keep.append(m)
                continue
            dist = rules.near_extreme_dist(st[0], st[1])
            if dist <= cfg.near_high_max_pct:
                keep.append(m)
            else:
                dropped[m["symbol"]] = (f"{dist * 100:.1f}% from {cfg.near_high_lookback}d closing high "
                                        f"(limit {cfg.near_high_max_pct * 100:.0f}%)")
        rows = keep

    # 6. recent trend: A (recent RS) + B (EMA21 slope)
    if rows and (cfg.recent_rs_filter or cfg.ema21_rising_filter):
        b_recent = bench_recent if cfg.recent_rs_filter else None
        keep = []
        for m in rows:
            why = []
            if cfg.recent_rs_filter and m["ret_recent"] is not None and b_recent is not None:
                if not (m["ret_recent"] - b_recent >= cfg.recent_rs_min_pct):
                    why.append(f"{cfg.recent_rs_days}d return {m['ret_recent']:+.1f}% vs benchmark {b_recent:+.1f}%")
            if cfg.ema21_rising_filter and m["slope"] is not None:
                if not (m["slope"] >= cfg.ema21_slope_min_pct):
                    why.append(f"EMA21 {m['slope']:+.1f}% over {cfg.ema21_slope_days}d")
            if why:
                dropped[m["symbol"]] = "; ".join(why)
            else:
                keep.append(m)
        rows = keep

    # 7. lower-high
    if rows and cfg.lower_high_filter:
        keep = []
        for m in rows:
            if m["lh"] is None or not rules.lower_high_fails(m["lh"]):
                keep.append(m)
            else:
                dropped[m["symbol"]] = "making lower highs"
        rows = keep

    out = [Selection(True, rules.LONG, m.get("rs"), m["adr"], m["turnover"], "selected", m["e9"], m["e21"], m["e50"],
                     m["symbol"]) for m in rows]
    return out, dropped


def select_symbol(D: Frame, kd: int, x: float, B: Optional[Frame], kb: int, cfg: StrategyConfig,
                  symbol: str = "") -> Selection:
    """One-stock wrapper around `build_watchlist` (no cross-stock cap applies to a single stock)."""
    out, dropped = build_watchlist([Candidate(symbol, D, kd, x)], B, kb, cfg)
    if out:
        return out[0]
    m = _metrics(Candidate(symbol, D, kd, x), cfg) or {}
    bench_rs, _ = _bench_returns(B, kb, cfg)
    rs = None
    if cfg.rs_enabled and m.get("raw") is not None:
        rs = m["raw"] - (bench_rs or 0.0)
    return Selection(False, None, rs, m.get("adr"), m.get("turnover", 0.0), dropped.get(symbol, "dropped"),
                     m.get("e9"), m.get("e21"), m.get("e50"), symbol)


def longs_allowed(B: Optional[Frame], kb: int, cfg: StrategyConfig, previous: Optional[bool]) -> Optional[bool]:
    """`_update_market_regime` for a long-only book: benchmark close above its EMA21 -> True (longs allowed),
    below -> False (longs blocked).  None when the filter is off; keeps `previous` inside the buffer or when
    data is missing."""
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
        return True
    if close < ema * (1 - buf):
        return False
    return previous
