"""Metrics + CSV/JSON output for a simulation `Result`."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .engine import Result


def _ts(ms: int) -> str:
    return pd.Timestamp(ms * 1_000_000).strftime("%Y-%m-%d %H:%M")


def trades_frame(res: Result) -> pd.DataFrame:
    rows = []
    for t in res.trades:
        d = asdict(t)
        d["net_pnl"] = t.net_pnl
        d["return_pct"] = (t.net_pnl / t.size_inr * 100.0) if t.size_inr else 0.0
        d["entry_ts"], d["exit_ts"] = _ts(t.entry_ts), _ts(t.exit_ts)
        rows.append(d)
    return pd.DataFrame(rows)


def fills_frame(res: Result) -> pd.DataFrame:
    rows = []
    for f in res.fills:
        d = asdict(f)
        d["ts"] = _ts(f.ts)
        rows.append(d)
    return pd.DataFrame(rows)


def equity_frame(res: Result) -> pd.DataFrame:
    return pd.DataFrame({"ts": [_ts(t) for t in res.ts], "equity": res.equity})


def _group(trades, key) -> Dict[str, dict]:
    g = defaultdict(lambda: {"n": 0, "net_pnl": 0.0, "wins": 0})
    for t in trades:
        k = key(t)
        g[k]["n"] += 1
        g[k]["net_pnl"] += t.net_pnl
        g[k]["wins"] += 1 if t.net_pnl > 0 else 0
    return {k: {**v, "net_pnl": round(v["net_pnl"], 2),
                "win_rate": round(v["wins"] / v["n"], 4)} for k, v in sorted(g.items())}


def _reason_bucket(reason: str) -> str:
    r = reason.lower()
    if r == "stop":
        return "STOP"
    if r in ("tp1", "tp2"):
        return r.upper()
    if "closes" in r and "without recovering" in r:
        return "time-limit exit"
    if "close while" in r:
        return "lower-close exit"
    if "flip" in r:
        return "ribbon flip"
    return reason


def summarize(res: Result) -> dict:
    exe, trades = res.exe, res.trades
    eq = pd.Series(res.equity, index=[pd.Timestamp(t * 1_000_000) for t in res.ts], dtype=float)
    out: dict = {"capital": exe.capital, "bars": len(res.ts)}
    if eq.empty:
        return {**out, "note": "no data"}
    daily = eq.groupby(eq.index.normalize()).last()
    rets = daily.pct_change().dropna()
    years = max((daily.index[-1] - daily.index[0]).days / 365.25, 1e-9)
    final = float(daily.iloc[-1])
    peak = daily.cummax()
    dd = (daily / peak - 1.0)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gp, gl = sum(t.net_pnl for t in wins), -sum(t.net_pnl for t in losses)
    out.update({
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / exe.capital - 1) * 100, 2),
        "cagr_pct": round(((final / exe.capital) ** (1 / years) - 1) * 100, 2) if final > 0 else None,
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "sharpe": round(float(rets.mean() / rets.std() * math.sqrt(252)), 2) if len(rets) > 2 and rets.std() > 0 else None,
        "closed_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 4) if trades else None,
        "profit_factor": round(gp / gl, 2) if gl > 0 else None,
        "avg_win_inr": round(gp / len(wins), 2) if wins else None,
        "avg_loss_inr": round(-gl / len(losses), 2) if losses else None,
        "net_pnl_closed_inr": round(sum(t.net_pnl for t in trades), 2),
        "fees_inr": round(sum(t.fees for t in trades), 2),
        "open_positions_at_end": len(res.open_positions),
        "unrealized_at_end_inr": round(sum(p["unrealized"] for p in res.open_positions), 2),
        "entries_rejected_no_margin": res.rejected_no_margin,
        "by_direction": _group(trades, lambda t: t.direction),
        "by_exit": _group(trades, lambda t: _reason_bucket(t.reason)),
        "by_symbol_top10": dict(sorted(_group(trades, lambda t: t.symbol).items(),
                                       key=lambda kv: -kv[1]["net_pnl"])[:10]),
        "watchlist_symbols": len(res.watch_days),
    })
    return out


def write_all(res: Result, out_dir: str) -> dict:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    summary = summarize(res)
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    trades_frame(res).to_csv(d / "trades.csv", index=False)
    fills_frame(res).to_csv(d / "fills.csv", index=False)
    equity_frame(res).to_csv(d / "equity.csv", index=False)
    pd.DataFrame(sorted(res.watch_days.items(), key=lambda kv: -kv[1]),
                 columns=["symbol", "bars_on_watchlist"]).to_csv(d / "watchlist_days.csv", index=False)
    if res.open_positions:
        pd.DataFrame([{**p, "stops": json.dumps(p["stops"]), "tp1": json.dumps(p["tp1"]), "tp2": json.dumps(p["tp2"]),
                       "entry_ts": _ts(p["entry_ts"])} for p in res.open_positions]).to_csv(d / "open_positions.csv", index=False)
    return summary
