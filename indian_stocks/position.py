"""Position-management rules - stops, take-profits and scale-ins (pure functions, no simulation).

Port of the arithmetic in `_on_entry_limit_filled`, `_place_tp_for_position`, `_tp2_target`,
`check_tp_filled`, `_on_additional_filled`, `_add_size_pct` and the gates of `_maintain_additional_limit`.
`side` is "LONG" or "SHORT"; prices in INR.

    initial stop   entry x (1 -/+ pct) for every tier in `stop_tiers` (default one tier: -10%, whole position);
                   tiers are independent stops, each covering its fraction of the entry size
    TP1            sell tp_size_pct (30%) of the position at avg x (1 +/- 20%); re-placed at the new average
                   after every add while TP1 has not filled
    after TP1      the runner's stop moves to the average entry (breakeven) when tp_stop_to_breakeven
    TP2            only after TP1: sell ema50_tp_size_pct (50%) of what REMAINS at
                   daily EMA50 x (1 +/- ema50_tp_mult x ADR%/100)   (EMA50 / ADR from completed daily candles)
    scale-in       one add per day, never on the entry day, max_additional_adds, only when the position is
                   >= pyramid_min_profit_pct (20%) in profit vs its average entry; size =
                   initial_value x (50%, 40%, 30%, 20%, 10%, 10% ...)
    after an add   tiers end -> ONE stop at new_avg x (1 -/+ stop_loss_pct[band]) (3%), or at new_avg if
                   stop_to_new_avg_after_add
    after a stop   no re-entry in that stock for the rest of the day
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from . import rules
from .config import StrategyConfig

LONG, SHORT = rules.LONG, rules.SHORT


def _sign(side: str) -> int:
    return 1 if side == LONG else -1


def _tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10) if tick > 0 else price


def _tiers(cfg: StrategyConfig) -> List[Tuple[float, float]]:
    tiers = sorted((float(p), float(f)) for p, f in cfg.stop_tiers if p > 0 and f > 0)
    return tiers or [(cfg.stop_loss_pct_ema9, 1.0)]


def stop_levels(side: str, entry: float, units: int, cfg: StrategyConfig) -> List[Tuple[float, int]]:
    """Initial stops as [(trigger_price, units)], shallowest tier first.  A single tier covers the
    whole position."""
    tiers = _tiers(cfg)
    if len(tiers) == 1:
        return [(_tick(entry * (1 - _sign(side) * tiers[0][0]), cfg.tick_size), units)]
    out = []
    for pct, frac in tiers:
        u = int(units * frac)
        if u > 0:
            out.append((_tick(entry * (1 - _sign(side) * pct), cfg.tick_size), u))
    return out


def stop_pct_for_band(band: str, cfg: StrategyConfig) -> float:
    return {"EMA9": cfg.stop_loss_pct_ema9, "EMA21": cfg.stop_loss_pct_ema21,
            "EMA50": cfg.stop_loss_pct_ema50}.get(band, cfg.stop_loss_pct_ema9)


def tp1(side: str, avg_entry: float, units: int, cfg: StrategyConfig) -> Optional[Tuple[float, int]]:
    """(price, units) of the first take-profit, or None if disabled / too small to sell a share."""
    if cfg.tp_profit_pct <= 0 or cfg.tp_size_pct <= 0:
        return None
    u = int(units * cfg.tp_size_pct)
    if u < 1:
        return None
    return _tick(avg_entry * (1 + _sign(side) * cfg.tp_profit_pct), cfg.tick_size), u


def stop_after_tp1(side: str, avg_entry: float, current_stop: float, cfg: StrategyConfig) -> float:
    """Breakeven for the runner (or the unchanged stop price when tp_stop_to_breakeven is off)."""
    return _tick(avg_entry, cfg.tick_size) if cfg.tp_stop_to_breakeven else current_stop


def tp2_target(side: str, ema50_daily: Optional[float], adr_percent: Optional[float],
               cfg: StrategyConfig) -> Optional[float]:
    """daily EMA50 x (1 +/- k x ADR%/100); None if disabled or inputs are missing."""
    if cfg.ema50_tp_mult <= 0 or cfg.ema50_tp_size_pct <= 0 or ema50_daily is None or adr_percent is None:
        return None
    target = ema50_daily * (1.0 + _sign(side) * cfg.ema50_tp_mult * adr_percent / 100.0)
    return _tick(target, cfg.tick_size) if target > 0 else None


def tp2_units(units_remaining: int, cfg: StrategyConfig) -> int:
    return int(units_remaining * cfg.ema50_tp_size_pct)


def stop_after_add(side: str, new_avg: float, band: str, cfg: StrategyConfig) -> float:
    """The single stop that replaces the tiers after a scale-in fills."""
    if cfg.stop_to_new_avg_after_add:
        return _tick(new_avg, cfg.tick_size)
    return _tick(new_avg * (1 - _sign(side) * stop_pct_for_band(band, cfg)), cfg.tick_size)


def profit_frac(side: str, avg_entry: float, live: float) -> float:
    return (live - avg_entry) / avg_entry if side == LONG else (avg_entry - live) / avg_entry


def add_allowed(side: str, avg_entry: float, live: float, adds_done: int, added_today: bool,
                cfg: StrategyConfig) -> Tuple[bool, str]:
    """The position-level gates for a scale-in.  (The setup gates - ribbon, RS lock, daily ribbon, close
    beyond the pyramid EMA, don't-chase - are `entry.select_setup` + `rules.*`.)"""
    if added_today:
        return False, "max one add per day (also blocks the entry day)"
    if adds_done >= cfg.max_additional_adds:
        return False, f"max {cfg.max_additional_adds} adds reached"
    prof = profit_frac(side, avg_entry, live)
    if prof < cfg.pyramid_min_profit_pct:
        return False, f"up {prof * 100:.1f}% < required {cfg.pyramid_min_profit_pct * 100:.0f}%"
    return True, ""


def add_inr(initial_value: float, adds_done: int, cfg: StrategyConfig) -> float:
    return initial_value * rules.add_size_pct(cfg.additional_entry_size_pct, cfg.additional_entry_size_step_pct,
                                              cfg.additional_entry_size_min_pct, adds_done)


def new_average(avg: float, units: int, add_price: float, add_units: int) -> float:
    return (avg * units + add_price * add_units) / (units + add_units)
