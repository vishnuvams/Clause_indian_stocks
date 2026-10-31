# Clause_indian_stocks

The **EMA 9/21/50 ribbon-pullback strategy** for **Indian (NSE) stocks**: stock selection, entry, exit, stops,
take-profits and scale-ins, plus the NSE stock list, data download and a daily scanner. **There is no backtester**
(and no broker code).

```bash
pip install -r requirements.txt

python -m indian_stocks universe --universe nifty500     # candidate list (NSE index CSV)
python -m indian_stocks fetch    --universe nifty500     # download + cache daily candles
python -m indian_stocks scan     --universe nifty500     # today's watchlist + entry limits, stops, TP1

python -m pytest                                         # 64 tests, no network needed
```

Ported from
[`ema_9_21_50_stocks_crypto_1day_1hour.py`](https://github.com/vishnuvams/clude_crypto/blob/claude/1h-ribbon-entry-exit/ema_9_21_50_stocks_crypto_1day_1hour.py)
(`vishnuvams/clude_crypto`, branch `claude/1h-ribbon-entry-exit`, commit `e295179`), using its
`VENUE_CFG["stocks"]` values and module constants.

## Layout

| File | What it holds | Original it mirrors |
|---|---|---|
| `indian_stocks/config.py` | every parameter (`StrategyConfig`) | module constants + `VENUE_CFG["stocks"]` |
| `indian_stocks/rules.py` | pure rule functions | `calculate_ema`, `_weak_run_decision`, `_check_exit`, `_raw_setup`, `_closed_candle_confirms`, `_daily_ribbon_confirms`, `_would_fill_immediately`, near-high / extension / lower-high helpers, `_add_size_pct` |
| `indian_stocks/selection.py` | the watchlist filters + benchmark regime | `_refresh_stocks`, `_update_market_regime` |
| `indian_stocks/entry.py` | the entry gate stack | `_select_setup`, `_maintain_entry_limit` |
| `indian_stocks/frames.py` | candle frame with EMA series ("done" vs "live" EMA) | `calculate_ema` on completed / in-progress candles |
| `indian_stocks/universe.py` | NSE stock list (Nifty 50/100/200/500, all EQ), ETF/fund/REIT filtering | `_is_equity` |
| `indian_stocks/data.py` | Yahoo, NSE bhavcopy and CSV candle providers, session-bar helpers | candle fetching |
| `indian_stocks/cli.py` | `universe`, `fetch`, `scan` | — |

```python
from indian_stocks.config import StrategyConfig
from indian_stocks.frames import Frame
from indian_stocks.selection import select_symbol
from indian_stocks.entry import evaluate_entry

D = Frame.from_df(daily_df, None)          # daily OHLCV + turnover (INR), index = trading date
B = Frame.from_df(nifty50_daily_df, None)  # benchmark
sel = select_symbol(D, kd=len(D) - 1, x=live_price, B=B, kb=len(B) - 1, cfg=StrategyConfig())
sel.ok, sel.direction, sel.reason          # on the watchlist? LONG leader? or which gate dropped it
```

## The rules

**Stock selection** — on the watchlist only if it passes, in order:

| # | Gate | Value |
|---|---|---|
| 1 | Volume floor — traded value of the last completed day | ≥ ₹4.5 crore (= the original's $500k × ₹90) |
| 2 | ADR floor — mean(high/low)−1 over 14 completed daily candles | ≥ 4 % |
| 3 | Relative strength — 30-day return minus Nifty 50's | ≥ +10 pts → **LONG leader** (≤ −10 pts → SHORT laggard, only if shorts are on). RS also **locks the side** for every later entry |
| 4 | Daily ribbon — live daily EMA9/21/50 monotonically stacked | either direction |
| 5 | Near-the-high — last close vs 30-day closing high | within 10 % |
| 6 | Recent trend — A: 10-day return beats the benchmark's · B: EMA21 up ≥ 1 % over 5 days | |
| 7 | Lower-high — highest close of days 1–5 ago not below that of days 6–10 ago | |

A stock that drops off the watchlist takes no new entries, but an open position in it is still managed.
Optional benchmark regime gate (Nifty 50 vs its EMA21) — off by default, as in the original.

**Entry** — rest a **limit on the live EMA9** only if: live EMA9 ≥ EMA21 ≥ EMA50 · RS side agrees · daily ribbon
agrees · the last completed close is above EMA9/21/50 · near-high ≤ 10 % · the 30-day closing high is ≤ 25 % above
the limit · no lower high · the live price is still *above* the limit (otherwise it would fill at market).
One entry per stock per day. Size ₹9 lakh long (= $10,000) / ₹2.25 lakh short (= $2,500).

**Exit** — evaluated once per completed candle:
1. **Lower-close rule on EMA9**, in profit *or* loss: once closes are below EMA9, exit on the first close *lower*
   than the previous one while still below; a close back above EMA9 cancels; 3 closes below with no lower close
   also exits.
2. **Ribbon flip** — EMA9 ≤ EMA21 (`exit_mode`; `ema9_ema50` and the full ribbon are options).

**Stops & take-profits** — hard stop **−10 %** from the fill · **TP1**: sell 30 % at +20 %, then the runner's stop
moves to breakeven · **TP2** (after TP1): sell 50 % of the remainder at `daily EMA50 × (1 + 7 × ADR)` ·
a stop blocks re-entry for the rest of the day.

**Scale-in** — from the next day, one add per day, only when **≥ +20 %** in profit, limit on EMA9 (same gates),
sized 50 → 40 → 30 → 20 → 10 % (floor) of the initial position, max 10; after an add the stop becomes **one stop
3 % under the new average**.

## Differences from the original

| Original (Hyperliquid stock perps) | Here |
|---|---|
| USD sizes / USD volume floor | INR at `USD_INR = 90` (`config.py`) |
| 5× leverage | `leverage = 1.0` (NSE cash delivery) |
| `ENABLE_SHORTS = False` | same, **off** — NSE cash cannot hold an overnight short. The mirror logic is implemented and tested |
| S&P 500 benchmark | Nifty 50 (you supply the series) |
| 24/7 candles | NSE session candles (09:15–15:30 IST) in `frames.py` |
| Shipped `signal_interval = "1d"` (file/branch name says 1h) | `1d` default, kept as a config field |

`frames.py` computes EMAs over the full history; the original recomputes over its last 250 candles. They agree to
within a rounding error once ~100 candles exist (and exactly when given the same input — see the parity tests).

## Data

| `--provider` | Source | Notes |
|---|---|---|
| `yahoo` (default) | `yfinance`, `<SYMBOL>.NS`, `^NSEI` | split/dividend **adjusted**; turnover ≈ close × volume |
| `nse` | NSE full bhavcopy `sec_bhavdata_full_DDMMYYYY.csv` | official, exact traded value, daily only, **unadjusted** (splits/bonuses look like gaps) |
| `csv` | your own `<SYMBOL>.csv` (`date,open,high,low,close,volume`) | offline / bring-your-own data |

The stock list comes from NSE's index CSVs or `EQUITY_L.csv` (`--universe all`). Only ordinary EQ-series shares are
kept (ETFs, index funds, InvITs, REITs, T2T/SME series are dropped — a heuristic; use `--blacklist` for strays).
If NSE is unreachable a short bundled starter list (`data/universe_fallback.csv`, 31 large caps — **not** real
index membership) is used and a warning is printed.

`scan` shows what the bot would do at the **next open if price opened where it last closed**: which stocks are on
the watchlist (and, for the rest, which gate dropped them), and for each watchlist stock whether an entry limit
would be rested, with its limit, −10 % stop and TP1.

> **Verification status.** The Yahoo and NSE download paths were written against their documented formats and
> tested with mocked responses only — the sandbox this was built in blocks `nseindia.com` and Yahoo. Run
> `python -m indian_stocks fetch --max-symbols 5` once with internet access to confirm them. Everything else is
> tested offline. The **ADR ≥ 4 % floor is high for Indian stocks** (large caps typically run well under it), so
> expect the watchlist to skew to volatile mid/small caps; lower it with `--adr-min`.

## Tests

`python -m pytest` — 64 tests.

* **Parity with the original** (`tests/test_parity.py`): the original bot's `calculate_ema`, `_weak_run_decision`,
  `_raw_setup`, `_closed_candle_confirms`, `_daily_ribbon_confirms` and the whole `_check_exit` ladder are
  extracted from its source with `ast` and run next to the ports on thousands of random inputs. Set `EMA_BOT_PATH`
  to the original file (the module is skipped if it is absent).
* Rule examples, every selection gate, and every entry gate in isolation.
* Data parsing (bhavcopy, Yahoo normalisation/caching with mocked downloads, session-bar resampling, incomplete
  last bar), universe filtering, and the CLI end-to-end on CSV files.

## What is not here

* **A backtester** — deliberately left out. (An earlier commit, `1c93ccc`, had one; it was removed on request.)
* Broker integration / live order placement, alerts, trade logging, restart-sync.
