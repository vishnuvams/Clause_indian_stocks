"""Entry gates (long-only), used by the `scan` command.

Port of `_select_setup` + `_maintain_entry_limit` (+ the reason text of `_flat_reason`).

An entry limit is rested only if ALL of these hold, in this order:
    1. one entry per symbol per day, symbol is on the watchlist
    2. LIVE signal-timeframe ribbon EMA9 >= EMA21 >= EMA50
    3. regime gate (benchmark vs its EMA21) - only if enabled
    4. DAILY ribbon agrees (EMA9 > EMA21 > EMA50)
    5. the last COMPLETED candle closed above the band EMA (EMA9) and every slower EMA
    6. near-the-high  : last daily close within 10% of the 30d closing high
    7. limit-extension: the 30d closing high is within 25% above the limit (the EMA9 the buy rests on)
    8. lower-high     : not making lower highs
    9. don't chase    : live price must still be ABOVE the limit - otherwise it'd fill at market
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import rules
from .config import StrategyConfig


def select_setup(cfg: StrategyConfig, live, band: str, regime_blocks_longs: bool,
                 daily_ribbon) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """(direction, band, limit_price, reason).  reason is '' when a setup exists."""
    direction, band, price = rules.raw_setup(*live, band)
    if band is None:
        return None, None, None, "ribbon not stacked (live EMA9 >= EMA21 >= EMA50 needed)"
    if regime_blocks_longs:
        return None, None, None, "regime blocks longs"
    if cfg.daily_ribbon_confirm and not rules.daily_ribbon_confirms(*daily_ribbon):
        return None, None, None, "daily ribbon does not confirm"
    return direction, band, price, ""


def evaluate_entry(cfg: StrategyConfig, *, live, done, prev_close, daily_ribbon, near, lower_high,
                   x: float, regime_blocks_longs: bool, band: str) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """Run gates 2-9.  Returns (direction, band, limit, reason); direction None => no entry."""
    direction, band, limit, why = select_setup(cfg, live, band, regime_blocks_longs, daily_ribbon)
    if band is None:
        return None, None, None, why
    if not rules.closed_candle_confirms(prev_close, *done, band):
        return None, None, None, f"last closed candle {prev_close:.2f} is not above the {band}+ ribbon"
    if cfg.near_high_max_pct > 0 and near:
        dist = rules.near_extreme_dist(near[0], near[1])
        if dist > cfg.near_high_max_pct:
            return None, None, None, (f"{dist * 100:.1f}% below the {near[2]}d closing high "
                                      f"(limit {cfg.near_high_max_pct * 100:.0f}%)")
    if cfg.limit_extension_max_pct > 0 and near and limit and limit > 0:
        ext = rules.limit_extension_pct(near[1], limit)
        if ext > cfg.limit_extension_max_pct:
            return None, None, None, (f"30d closing high is {ext * 100:.1f}% above the entry limit "
                                      f"{limit:.2f} (limit {cfg.limit_extension_max_pct * 100:.0f}%)")
    if cfg.lower_high_filter and lower_high and rules.lower_high_fails(lower_high):
        return None, None, None, "making lower highs"
    if rules.would_fill_immediately(limit, x):
        return None, None, None, f"price {x:.2f} already past limit {limit:.2f} (would fill at market)"
    return direction, band, limit, ""
