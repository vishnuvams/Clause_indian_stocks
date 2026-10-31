"""Stock selection - port of `EMAPullbackBot._refresh_stocks` (+ `_update_market_regime`), stage for stage.

`build_watchlist` takes the whole candidate set and returns the ordered watchlist exactly as the original
builds it (same stages, same order of stages, same tie-breaking):

    1. volume floor       traded value of the last completed day >= min_turnover_inr
    2. ADR floor          mean(high/low)-1 over the last 14 completed daily candles >= 4 %    (no data => dropped)
    3. relative strength  rs = 30d return - benchmark's 30d return (raw return if no benchmark data)
                          ranked high -> low;  LONG leaders: rs >= +10 pts (top `rs_top_n_long` if > 0)
                          SHORT laggards (only if shorts on): rs <= -10 pts (bottom `rs_top_n_short` if > 0);
                          a stock that is both is a leader.  The cap is applied HERE, i.e. to the ADR
                          survivors, BEFORE the filters below - so fewer than N can survive.
                          RS off: no side, ordered by traded value.
    4. daily ribbon       live daily EMA9/21/50 monotonically stacked (either way)
    5. near-the-high      last completed close within 10 % of its 30d closing high (SHORT: low)
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
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

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

    def sel(m, **kw):
        return Selection(True, kw.get("direction"), m.get("rs"), m["adr"], m["turnover"], "selected",
                         m["e9"], m["e21"], m["e50"], m["symbol"])

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

    # 3. relative strength -> side (+ ranking, caps)
    side: Dict[str, Optional[str]] = {}
    bench_rs, bench_recent = _bench_returns(B, kb, cfg)
    if cfg.rs_enabled and rows:
        for m in rows:
            m["rs"] = None if m["raw"] is None else (m["raw"] if bench_rs is None else m["raw"] - bench_rs)
        for m in rows:
            if m["rs"] is None:
                dropped[m["symbol"]] = "not enough history for RS"
        ranked = sorted((m for m in rows if m["rs"] is not None), key=lambda m: m["rs"], reverse=True)
        thr_l, thr_s = cfg.rs_min_pct_long / 100.0, cfg.rs_min_pct_short / 100.0
        long_ranked = [m for m in ranked if m["rs"] >= thr_l]
        short_ranked = [m for m in ranked if m["rs"] <= -thr_s]
        leaders = long_ranked[:cfg.rs_top_n_long] if cfg.rs_top_n_long > 0 else long_ranked
        if cfg.enable_shorts:
            laggards = short_ranked[-cfg.rs_top_n_short:] if cfg.rs_top_n_short > 0 else short_ranked
        else:
            laggards = []
        leader_syms = {m["symbol"] for m in leaders}
        laggard_syms = {m["symbol"] for m in laggards} - leader_syms
        side = {s: rules.LONG for s in leader_syms}
        side.update({s: rules.SHORT for s in laggard_syms})
        rows = [m for m in ranked if m["symbol"] in leader_syms or m["symbol"] in laggard_syms]
        for m in ranked:
            if m["symbol"] not in side:
                dropped[m["symbol"]] = (
                    f"RS {m['rs'] * 100:+.1f} pts not beyond +{cfg.rs_min_pct_long:.0f}"
                    + (f" / -{cfg.rs_min_pct_short:.0f}" if cfg.enable_shorts else "")
                    if (m["rs"] < thr_l and not (cfg.enable_shorts and m["rs"] <= -thr_s))
                    else "cut by the rs_top_n cap")
    else:
        for m in rows:
            m["rs"] = None
            side[m["symbol"]] = None
        rows.sort(key=lambda m: m["turnover"], reverse=True)

    # 4. daily ribbon
    keep = []
    for m in rows:
        if m["ribbon_ok"]:
            keep.append(m)
        else:
            dropped[m["symbol"]] = "daily EMA ribbon not stacked"
    rows = keep

    def direction_of(m):
        d = side.get(m["symbol"])
        return d or (rules.LONG if (m["e9"] is not None and m["e21"] is not None and m["e9"] >= m["e21"]) else rules.SHORT)

    # 5. near-the-high
    if cfg.near_high_max_pct > 0:
        keep = []
        for m in rows:
            st = m["near"]
            if st is None:
                keep.append(m)
                continue
            d = direction_of(m)
            dist = rules.near_extreme_dist(d, st[0], st[1], st[2])
            if dist <= cfg.near_high_max_pct:
                keep.append(m)
            else:
                dropped[m["symbol"]] = (f"{dist * 100:.1f}% from {cfg.near_high_lookback}d closing "
                                        f"{'high' if d == rules.LONG else 'low'} (limit {cfg.near_high_max_pct * 100:.0f}%)")
        rows = keep

    # 6. recent trend: A (recent RS) + B (EMA21 slope)
    if rows and (cfg.recent_rs_filter or cfg.ema21_rising_filter):
        b_recent = bench_recent if cfg.recent_rs_filter else None
        keep = []
        for m in rows:
            d = direction_of(m)
            is_long = d == rules.LONG
            why = []
            if cfg.recent_rs_filter and m["ret_recent"] is not None and b_recent is not None:
                rel = m["ret_recent"] - b_recent
                ok = (rel >= cfg.recent_rs_min_pct) if is_long else (rel <= -cfg.recent_rs_min_pct)
                if not ok:
                    why.append(f"{cfg.recent_rs_days}d return {m['ret_recent']:+.1f}% vs benchmark {b_recent:+.1f}%")
            if cfg.ema21_rising_filter and m["slope"] is not None:
                ok = (m["slope"] >= cfg.ema21_slope_min_pct) if is_long else (m["slope"] <= -cfg.ema21_slope_min_pct)
                if not ok:
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
            d = direction_of(m)
            if m["lh"] is None or not rules.lower_high_fails(d, m["lh"]):
                keep.append(m)
            else:
                dropped[m["symbol"]] = "making lower highs" if d == rules.LONG else "making higher lows"
        rows = keep

    return [sel(m, direction=side.get(m["symbol"])) for m in rows], dropped


def select_symbol(D: Frame, kd: int, x: float, B: Optional[Frame], kb: int, cfg: StrategyConfig,
                  symbol: str = "") -> Selection:
    """One-stock wrapper around `build_watchlist` (no cross-stock caps apply to a single stock)."""
    out, dropped = build_watchlist([Candidate(symbol, D, kd, x)], B, kb, cfg)
    if out:
        return out[0]
    # the dropped reason; recompute cheaply for the numbers a caller may want
    m = _metrics(Candidate(symbol, D, kd, x), cfg) or {}
    bench_rs, _ = _bench_returns(B, kb, cfg)
    rs = None
    if cfg.rs_enabled and m.get("raw") is not None:
        rs = m["raw"] - (bench_rs or 0.0)
    return Selection(False, None, rs, m.get("adr"), m.get("turnover", 0.0), dropped.get(symbol, "dropped"),
                     m.get("e9"), m.get("e21"), m.get("e50"), symbol)


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
