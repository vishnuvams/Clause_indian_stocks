"""Configuration - the EMA 9/21/50 ribbon-pullback strategy applied to NSE stocks.

Source of truth: `ema_9_21_50_stocks_crypto_1day_1hour.py` (branch claude/1h-ribbon-entry-exit of
vishnuvams/clude_crypto).  Every number below is that file's module constant or its
`VENUE_CFG["stocks"]` value.  Only things that cannot exist on NSE are re-expressed:

    USD sizes / USD volume floor  ->  INR (1 USD ~ 90 INR)
    HIP-3 leverage 5x             ->  1x (NSE cash delivery); `leverage` is still a knob
    SP500 benchmark               ->  Nifty 50
    24/7 candles                  ->  NSE session candles (09:15-15:30 IST)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple

USD_INR = 90.0   # only used to convert the original's USD figures below; change if you like


@dataclass
class StrategyConfig:
    # ── Timeframes ──────────────────────────────────────────────────────────
    signal_interval: str = "1d"       # entry/exit ribbon timeframe (VENUE_CFG signal_interval)
    pyramid_interval: str = "1d"      # timeframe the scale-in add is evaluated on

    # ── Indicators ──────────────────────────────────────────────────────────
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50

    # ── Direction ───────────────────────────────────────────────────────────
    regime_filter_enabled: bool = False
    regime_ema_period: int = 21       # REGIME_EMA_PERIOD
    regime_buffer_pct: float = 0.0    # REGIME_BUFFER_PCT

    # ── Universe / selection (all on DAILY candles) ─────────────────────────
    min_turnover_inr: float = 500_000 * USD_INR   # MIN_VOLUME_USD = 500k  -> ~4.5 crore INR
    adr_period: int = 14              # ADR_PERIOD
    adr_min_pct: float = 4.0          # ADR_MIN_PCT
    rs_enabled: bool = True           # RS_ENABLED
    rs_lookback_days: int = 30        # RS_LOOKBACK_DAYS
    rs_min_pct_long: float = 10.0     # stocks: must beat the benchmark by >= this many points
    rs_top_n_long: int = 0            # 0 = no cap
    recent_rs_filter: bool = True     # A
    recent_rs_days: int = 10          # RECENT_RS_DAYS
    recent_rs_min_pct: float = 0.0    # RECENT_RS_MIN_PCT
    ema21_rising_filter: bool = True  # B
    ema21_slope_days: int = 5         # EMA21_SLOPE_DAYS
    ema21_slope_min_pct: float = 1.0  # EMA21_SLOPE_MIN_PCT
    lower_high_filter: bool = True
    lower_high_recent_days: int = 5   # LOWER_HIGH_RECENT_DAYS
    lower_high_prior_days: int = 5    # LOWER_HIGH_PRIOR_DAYS   (NB: the old backtester used 10)
    near_high_max_pct: float = 0.10   # stocks: 10%  (crypto: 20%); 0 = off
    near_high_lookback: int = 30      # NEAR_HIGH_LOOKBACK_DAYS
    limit_extension_max_pct: float = 0.25
    daily_ribbon_confirm: bool = True
    universe_refresh_hours: float = 2.0   # UNIVERSE_REFRESH_MIN = 120

    # ── Entry ───────────────────────────────────────────────────────────────
    entry_ema: str = "ema9"           # LONG limit sits here
    limit_refresh_hours: float = 1.0  # LIMIT_REFRESH_HOURS
    initial_entry_inr_long: float = 10_000 * USD_INR   # initial_entry_usd_long  = 10000
    leverage: float = 1.0

    # ── Scale-in (pyramid) ──────────────────────────────────────────────────
    pyramid_ema: str = "ema9"
    additional_entry_size_pct: float = 0.5
    additional_entry_size_step_pct: float = 0.10   # 50%, 40%, 30%, 20%, 10%, 10% ...
    additional_entry_size_min_pct: float = 0.10
    max_additional_adds: int = 10
    pyramid_min_profit_pct: float = 0.20           # PYRAMID_MIN_PROFIT_PCT
    stop_to_new_avg_after_add: bool = False

    # ── Exits (evaluated once per completed signal candle) ──────────────────
    exit_mode: str = "ema9_ema21"            # ema9_ema21_ema50 | ema9_ema50 | ema9_ema21
    exit_profit_2closes: str = "ema9"        # reference EMA of the lower-close rule: ema9|ema21|ema50|off
    exit_lower_close_max_closes: int = 3     # EXIT_LOWER_CLOSE_MAX_CLOSES (time limit)

    # ── Stops ───────────────────────────────────────────────────────────────
    stop_loss_pct_ema9: float = 0.03         # single stop used after an add, per entry band
    stop_loss_pct_ema21: float = 0.03
    stop_loss_pct_ema50: float = 0.03
    stop_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [(0.10, 1.0)])

    # ── Take profits ────────────────────────────────────────────────────────
    tp_profit_pct: float = 0.20              # TP_PROFIT_PCT
    tp_size_pct: float = 0.30                # TP_SIZE_PCT
    tp_stop_to_breakeven: bool = True
    ema50_tp_mult: float = 7.0               # EMA50_TP_MULT  (the old backtester defaulted to 5)
    ema50_tp_size_pct: float = 0.50
    ema50_tp_adr_period: int = 14

    # ── Window the live bot fetches (250 candles, the last one is in progress)
    window_candles: int = 250

    # ── NSE equity tick
    tick_size: float = 0.05

    def as_dict(self) -> dict:
        return asdict(self)
