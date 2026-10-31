"""Entry gates, shared by the simulator (`engine.py`) and the live `scan` command.

Port of `_select_setup` + `_maintain_entry_limit` (+ the reason text of `_flat_reason`).

An entry limit is rested only if ALL of these hold, in this order:
    1. one entry per symbol per day, symbol is on the watchlist
    2. LIVE signal-timeframe ribbon EMA9 >= EMA21 >= EMA50 (LONG; mirror for SHORT if enabled)
    3. regime gate (benchmark vs its EMA21) - only if enabled
    4. RS lock: a leader may only LONG, a laggard only SHORT
    5. DAILY ribbon agrees (EMA9 > EMA21 > EMA50)
    6. the last COMPLETED candle closed beyond the band EMA and every slower EMA
    7. near-the-high  : last daily close within 10% of the 30d closing high
    8. limit-extension: the 30d closing high is within 25% above the limit (the EMA9 the buy rests on)
    9. lower-high     : not making lower highs
   10. don't chase    : live price must still be ABOVE the limit (LONG) - otherwise it'd fill at market
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from . import rules
from .config import StrategyConfig


def select_setup(cfg: StrategyConfig, live, band_for: Callable[[str], str], regime: Optional[str],
                 rs_dir: Optional[str], daily_ribbon) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """(direction, band, limit_price, reason).  reason is '' when a setup exists."""
    direction, band, price = rules.raw_setup(*live, cfg.enable_shorts, band_for)
    if band is None:
        return None, None, None, "ribbon not stacked (live EMA9 >= EMA21 >= EMA50 needed)"
    if regime is not None and direction != regime:
        return None, None, None, f"regime blocks {direction}"
    if rs_dir is not None and direction != rs_dir:
        return None, None, None, f"relative strength allows {rs_dir} only"
    if cfg.daily_ribbon_confirm and not rules.daily_ribbon_confirms(direction, *daily_ribbon):
        return None, None, None, f"daily ribbon does not confirm {direction}"
    return direction, band, price, ""


def evaluate_entry(cfg: StrategyConfig, *, live, done, prev_close, daily_ribbon, near, lower_high,
                   x: float, rs_dir: Optional[str], regime: Optional[str],
                   eband: str, eband_s: str) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """Run gates 2-10.  Returns (direction, band, limit, reason); direction None => no entry."""
    direction, band, limit, why = select_setup(
        cfg, live, lambda d: eband_s if d == rules.SHORT else eband, regime, rs_dir, daily_ribbon)
    if band is None:
        return None, None, None, why
    if not rules.closed_candle_confirms(direction, prev_close, *done, band):
        return None, None, None, f"last closed candle {prev_close:.2f} is not beyond the {band}+ ribbon ({direction})"
    if cfg.near_high_max_pct > 0 and near:
        dist = rules.near_extreme_dist(direction, *near[:3])
        if dist > cfg.near_high_max_pct:
            return None, None, None, (f"{dist * 100:.1f}% from the {near[3]}d closing "
                                      f"{'high' if direction == rules.LONG else 'low'} (limit {cfg.near_high_max_pct * 100:.0f}%)")
    if cfg.limit_extension_max_pct > 0 and near and limit and limit > 0:
        ext = rules.limit_extension_pct(direction, near[1], near[2], limit)
        if ext > cfg.limit_extension_max_pct:
            return None, None, None, (f"30d closing extreme is {ext * 100:.1f}% beyond the entry limit "
                                      f"{limit:.2f} (limit {cfg.limit_extension_max_pct * 100:.0f}%)")
    if cfg.lower_high_filter and lower_high and rules.lower_high_fails(direction, lower_high):
        return None, None, None, "making lower highs" if direction == rules.LONG else "making higher lows"
    if rules.would_fill_immediately(direction, limit, x):
        return None, None, None, f"price {x:.2f} already past limit {limit:.2f} (would fill at market)"
    return direction, band, limit, ""
