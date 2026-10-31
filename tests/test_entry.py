"""Entry gates (entry.evaluate_entry) one at a time, from a passing baseline."""
import dataclasses

from indian_stocks.config import StrategyConfig
from indian_stocks.entry import evaluate_entry

CFG = StrategyConfig()
# bullish ribbon, last close above all EMAs, price above the EMA9 limit, 30d high close to the limit
BASE = dict(live=(110.0, 105.0, 100.0), done=(109.0, 104.5, 99.5), prev_close=112.0,
            daily_ribbon=(110.0, 105.0, 100.0), near=(115.0, 118.0, 90.0, 30), lower_high=(118.0, 116.0, 100.0, 98.0),
            x=113.0, rs_dir="LONG", regime=None, eband="EMA9", eband_s="EMA21")


def ev(cfg=CFG, **over):
    return evaluate_entry(cfg, **{**BASE, **over})


def test_baseline_places_a_long_limit_on_ema9():
    assert ev()[:3] == ("LONG", "EMA9", 110.0)


def test_each_gate_blocks():
    assert ev(live=(100.0, 105.0, 110.0))[0] is None                         # ribbon not stacked (long-only)
    assert ev(live=(110.0, 105.0, None))[0] is None                          # needs the full ribbon
    assert "relative strength" in ev(rs_dir="SHORT")[3]                      # RS locks the side
    assert ev(regime="SHORT")[0] is None                                     # regime gate
    assert ev(daily_ribbon=(100.0, 105.0, 110.0))[0] is None                 # daily ribbon disagrees
    assert ev(prev_close=108.0)[0] is None                                   # last close not above EMA9
    assert "closing high" in ev(near=(100.0, 118.0, 90.0, 30))[3]            # >10% below the 30d high
    assert "beyond the entry limit" in ev(near=(138.0, 140.0, 90.0, 30))[3]  # 30d high >25% above the limit
    assert "lower highs" in ev(lower_high=(110.0, 116.0, 100.0, 98.0))[3]
    assert "would fill at market" in ev(x=109.0)[3]                          # don't chase


def test_gates_can_be_switched_off():
    off = dataclasses.replace(CFG, near_high_max_pct=0.0, limit_extension_max_pct=0.0, lower_high_filter=False,
                              daily_ribbon_confirm=False)
    assert ev(off, near=(100.0, 140.0, 90.0, 30), lower_high=(1.0, 2.0, 1.0, 2.0),
              daily_ribbon=(1.0, 2.0, 3.0))[0] == "LONG"


def test_short_side_is_mirrored_and_sits_on_ema21():
    cfg = dataclasses.replace(CFG, enable_shorts=True)
    r = evaluate_entry(cfg, live=(90.0, 95.0, 100.0), done=(91.0, 95.5, 100.5), prev_close=88.0,
                       daily_ribbon=(90.0, 95.0, 100.0), near=(88.0, 110.0, 86.0, 30),
                       lower_high=(95.0, 97.0, 86.0, 88.0), x=92.0, rs_dir="SHORT", regime=None,
                       eband="EMA9", eband_s="EMA21")
    assert r[:3] == ("SHORT", "EMA21", 95.0)
