import dataclasses

import pytest

from indian_stocks import position as pos
from indian_stocks.config import StrategyConfig

CFG = StrategyConfig()


def test_single_tier_stop_is_10pct_for_the_whole_position():
    assert pos.stop_levels(500.0, 1800, CFG) == [(450.0, 1800)]


def test_multi_tier_stops_split_the_position():
    cfg = dataclasses.replace(CFG, stop_tiers=[(0.10, 0.34), (0.05, 0.33), (0.07, 0.33)])
    assert pos.stop_levels(100.0, 100, cfg) == [(95.0, 33), (93.0, 33), (90.0, 34)]


def test_tp1_is_30pct_of_units_at_plus_20pct():
    assert pos.tp1(500.0, 1801, CFG) == (600.0, 540)
    assert pos.tp1(500.0, 2, CFG) is None                                      # 30% of 2 shares < 1 share
    assert pos.tp1(500.0, 100, dataclasses.replace(CFG, tp_profit_pct=0)) is None


def test_breakeven_after_tp1_and_the_switch():
    assert pos.stop_after_tp1(500.0, 450.0, CFG) == 500.0
    assert pos.stop_after_tp1(500.0, 450.0, dataclasses.replace(CFG, tp_stop_to_breakeven=False)) == 450.0


def test_tp2_target_is_ema50_times_one_plus_7_adr():
    assert pos.tp2_target(1000.0, 5.0, CFG) == pytest.approx(1350.0)           # 1000 x (1 + 7 x 5%)
    assert pos.tp2_target(None, 5.0, CFG) is None and pos.tp2_target(1000.0, None, CFG) is None
    assert pos.tp2_target(1000.0, 5.0, dataclasses.replace(CFG, ema50_tp_mult=0)) is None
    assert pos.tp2_units(1000, CFG) == 500


def test_stop_after_add_is_3pct_under_the_new_average_or_breakeven():
    assert pos.stop_after_add(600.0, "EMA9", CFG) == 582.0
    assert pos.stop_after_add(600.0, "EMA9", dataclasses.replace(CFG, stop_to_new_avg_after_add=True)) == 600.0
    wide = dataclasses.replace(CFG, stop_loss_pct_ema50=0.06)
    assert pos.stop_after_add(600.0, "EMA50", wide) == 564.0


def test_add_gates():
    assert pos.add_allowed(100.0, 120.0, 0, False, CFG)[0]                     # exactly +20% passes
    assert not pos.add_allowed(100.0, 119.9, 0, False, CFG)[0]                 # +19.9%
    assert not pos.add_allowed(100.0, 150.0, 0, True, CFG)[0]                  # already added today / entry day
    assert not pos.add_allowed(100.0, 150.0, 10, False, CFG)[0]                # cap
    assert pos.add_allowed(100.0, 150.0, 9, False, CFG)[0]
    assert not pos.add_allowed(100.0, 90.0, 0, False, CFG)[0]                  # never average into a loser


def test_add_sizes_follow_50_40_30_20_10_10():
    got = [round(pos.add_inr(900_000, n, CFG)) for n in range(7)]
    assert got == [450_000, 360_000, 270_000, 180_000, 90_000, 90_000, 90_000]


def test_new_average():
    assert pos.new_average(100.0, 10, 130.0, 5) == pytest.approx(110.0)
