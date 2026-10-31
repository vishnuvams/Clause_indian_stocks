"""Command line:  python -m indian_stocks <universe|fetch|scan> ..."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Dict, List, Optional

import pandas as pd

from . import rules
from .config import StrategyConfig
from .data import drop_incomplete_last_bar, get_provider, ist_now
from .entry import evaluate_entry
from . import position as posrules
from .frames import Frame
from .selection import longs_allowed, select_symbol
from .universe import load_universe

WARMUP_DAYS = 420      # calendar days of history loaded before --start (EMA50 + 250-candle windows)


def _ist_today() -> pd.Timestamp:
    return pd.Timestamp(ist_now()).normalize()


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def add_common(ap: argparse.ArgumentParser):
    ap.add_argument("--universe", default="nifty500", help="nifty50|nifty100|nifty200|nifty500|all|file:<csv>")
    ap.add_argument("--symbols", nargs="*", help="explicit symbols (overrides --universe)")
    ap.add_argument("--max-symbols", type=int, default=0, help="use only the first N symbols (quick runs)")
    ap.add_argument("--blacklist", nargs="*", default=[])
    ap.add_argument("--provider", default="yahoo", choices=["yahoo", "nse", "csv"])
    ap.add_argument("--data-dir", help="folder of <SYMBOL>.csv files (--provider csv)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--offline", action="store_true", help="do not download the NSE universe list")
    ap.add_argument("--benchmark", default=None, help="default ^NSEI (yahoo) / NIFTYBEES (nse, csv)")


def add_strategy(ap: argparse.ArgumentParser):
    d = StrategyConfig()
    ap.add_argument("--min-turnover-cr", type=float, default=d.min_turnover_inr / 1e7,
                    help="volume floor in crore INR of traded value (default = 500k USD)")
    ap.add_argument("--adr-min", type=float, default=d.adr_min_pct)
    ap.add_argument("--rs-long", type=float, default=d.rs_min_pct_long)
    ap.add_argument("--near-high", type=float, default=d.near_high_max_pct)
    ap.add_argument("--exit-mode", default=d.exit_mode, choices=["ema9_ema21", "ema9_ema50", "ema9_ema21_ema50"])
    ap.add_argument("--lower-close-ema", default=d.exit_profit_2closes, choices=["ema9", "ema21", "ema50", "off"])
    ap.add_argument("--long-inr", type=float, default=d.initial_entry_inr_long)
    ap.add_argument("--leverage", type=float, default=d.leverage)
    ap.add_argument("--regime", type=_str2bool, default=d.regime_filter_enabled)
    ap.add_argument("--signal-interval", default=d.signal_interval)
    ap.add_argument("--pyramid-interval", default=d.pyramid_interval)


def make_cfg(a) -> StrategyConfig:
    return replace(StrategyConfig(),
                   min_turnover_inr=a.min_turnover_cr * 1e7, adr_min_pct=a.adr_min,
                   rs_min_pct_long=a.rs_long, near_high_max_pct=a.near_high,
                   exit_mode=a.exit_mode, exit_profit_2closes=a.lower_close_ema,
                   initial_entry_inr_long=a.long_inr, leverage=a.leverage,
                   regime_filter_enabled=a.regime, signal_interval=a.signal_interval,
                   pyramid_interval=a.pyramid_interval)


def pick_symbols(a) -> List[str]:
    if a.symbols:
        syms = [s.upper() for s in a.symbols]
    else:
        syms = load_universe(a.universe, cache_dir=a.cache_dir, blacklist=a.blacklist, offline=a.offline)["symbol"].tolist()
    return syms[:a.max_symbols] if a.max_symbols else syms


def benchmark_symbol(a) -> str:
    return a.benchmark or ("^NSEI" if a.provider == "yahoo" else "NIFTYBEES")


# ───────────────────────────── commands ─────────────────────────────

def cmd_universe(a):
    df = load_universe(a.universe, cache_dir=a.cache_dir, blacklist=a.blacklist, offline=a.offline)
    print(df.to_string(index=False))
    print(f"\n{len(df)} stocks in '{a.universe}' (ordinary EQ-series shares only)")


def _provider(a):
    return get_provider(a.provider, cache_dir=a.cache_dir, data_dir=a.data_dir)


def cmd_fetch(a):
    syms = pick_symbols(a)
    end = _ist_today()
    start = end - pd.Timedelta(days=a.days)
    prov = _provider(a)
    data = prov.fetch(syms + [benchmark_symbol(a)], start, end, "1d")
    print(f"fetched {len(data)}/{len(syms) + 1} symbols, "
          f"{sum(len(d) for d in data.values())} daily bars -> {a.cache_dir}")
    missing = [s for s in syms if s not in data]
    if missing:
        print(f"no data for {len(missing)} symbols: {', '.join(missing[:30])}{' ...' if len(missing) > 30 else ''}")


def cmd_scan(a):
    """What the bot would do at the NEXT open if price opened where it last closed."""
    cfg = make_cfg(a)
    bench_sym = benchmark_symbol(a)
    syms = pick_symbols(a)
    end = _ist_today()
    prov = _provider(a)
    data = prov.fetch(syms + [bench_sym], end - pd.Timedelta(days=WARMUP_DAYS), end, "1d")
    bench_df = data.pop(bench_sym, None)
    bench = Frame.from_df(drop_incomplete_last_bar(bench_df), None) if bench_df is not None and len(bench_df) else None
    kb = len(bench) - 1 if bench else -1
    longs_ok = longs_allowed(bench, kb, cfg, None)          # None = regime filter off
    rows = []
    for sym in syms:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        D = Frame.from_df(drop_incomplete_last_bar(df), None)
        kd = len(D) - 1
        x = D.c[kd]
        sel = select_symbol(D, kd, x, bench, kb, cfg)
        row = {"symbol": sym, "close": x, "selected": sel.ok,
               "rs_pts": None if sel.rs is None else round(sel.rs * 100, 1), "adr_pct": sel.adr and round(sel.adr, 2),
               "traded_cr": round(sel.turnover / 1e7, 2), "why": "" if sel.ok else sel.reason,
               "entry_limit": None, "stop": None, "tp1": None, "note": ""}
        if sel.ok:
            c = cfg
            live, done = D.triple_live(kd, x, (9, 21, 50)), D.triple_done(kd, (9, 21, 50))
            need = max(c.near_high_lookback, c.lower_high_recent_days + c.lower_high_prior_days) + 1
            closes = D.closes_done(kd, need)
            direction, band, limit, why = evaluate_entry(
                c, live=live, done=done, prev_close=D.c[kd], daily_ribbon=live,
                near=rules.near_extreme_stats(closes, c.near_high_lookback),
                lower_high=rules.lower_high_stats(closes, c.lower_high_recent_days, c.lower_high_prior_days),
                x=x, regime_blocks_longs=longs_ok is False, band=rules.band_label(c.entry_ema))
            if direction:
                units = max(int(c.initial_entry_inr_long / limit), 1)
                stop = posrules.stop_levels(limit, units, c)[-1][0]                  # deepest tier
                t1 = posrules.tp1(limit, units, c)
                row.update(entry_limit=round(limit, 2), stop=round(stop, 2),
                           tp1=round(t1[0], 2) if t1 else None,
                           note=f"limit on {band}")
            else:
                row["note"] = why
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("no data - nothing to scan")
    out = out.sort_values(["selected", "rs_pts"], ascending=[False, False])
    out.to_csv(a.out, index=False)
    sel = out[out["selected"]]
    pd.set_option("display.width", 200)
    print(f"regime: {'filter off' if longs_ok is None else ('longs allowed' if longs_ok else 'longs blocked')} | {len(sel)} of {len(out)} stocks on the watchlist\n")
    if len(sel):
        print(sel.drop(columns=["selected", "why"]).to_string(index=False))
    print(f"\nfull table (incl. why each stock was dropped) -> {a.out}")


def cmd_zerodha(a):
    from .zerodha_runner import ZerodhaRunner, kite_from_env
    cfg = make_cfg(a)
    runner = ZerodhaRunner(kite_from_env(), cfg, pick_symbols(a), live=a.live, adopt_existing=a.adopt_existing,
                           blacklist=a.blacklist, refresh_minutes=a.refresh_minutes)
    print(("LIVE - orders WILL be sent to Zerodha" if a.live else "DRY RUN - nothing is sent to Zerodha"))
    runner.cycle() if a.once else runner.run_forever(a.poll_minutes)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m indian_stocks",
                                 description="EMA 9/21/50 ribbon-pullback strategy on Indian (NSE) stocks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("universe", help="list the candidate stocks")
    add_common(u)
    u.set_defaults(fn=cmd_universe)

    f = sub.add_parser("fetch", help="download + cache daily candles")
    add_common(f)
    f.add_argument("--days", type=int, default=900)
    f.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("scan", help="today's watchlist + entry setups (what the bot would do at the next open)")
    add_common(s)
    add_strategy(s)
    s.add_argument("--out", default="results/scan.csv")
    s.set_defaults(fn=cmd_scan)


    z = sub.add_parser("zerodha", help="run the strategy on a Zerodha (Kite Connect) account - DRY RUN unless --live")
    add_common(z)
    add_strategy(z)
    z.add_argument("--live", action="store_true", help=f"send real orders (also needs {'ZERODHA_CONFIRM_LIVE'}=I_UNDERSTAND)")
    z.add_argument("--once", action="store_true", help="run a single cycle and exit")
    z.add_argument("--adopt-existing", action="store_true", help="manage stocks you already hold (dangerous)")
    z.add_argument("--poll-minutes", type=float, default=30)
    z.add_argument("--refresh-minutes", type=float, default=120)
    z.set_defaults(fn=cmd_zerodha)
    return ap


def main(argv: Optional[List[str]] = None):
    a = build_parser().parse_args(argv)
    a.fn(a)
