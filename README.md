# Clause_indian_stocks

The **EMA 9/21/50 ribbon-pullback strategy** applied to **Indian (NSE) stocks** — a data layer, a stock
selector, a portfolio backtester and a daily scanner.

The entry, exit and stock-selection rules are ported from
[`ema_9_21_50_stocks_crypto_1day_1hour.py`](https://github.com/vishnuvams/clude_crypto/blob/claude/1h-ribbon-entry-exit/ema_9_21_50_stocks_crypto_1day_1hour.py)
(`vishnuvams/clude_crypto`, branch `claude/1h-ribbon-entry-exit`, commit `e295179`), using its
`VENUE_CFG["stocks"]` values and module constants. Function names in `indian_stocks/rules.py` mirror the
original's so they can be diffed.

> **Scope.** This is research tooling: it selects stocks, generates the entry/exit logic and simulates it.
> It does **not** place orders with a broker. See [What is not here](#what-is-not-here).

```bash
pip install -r requirements.txt

python -m indian_stocks universe --universe nifty500          # candidate list (NSE index CSV)
python -m indian_stocks fetch    --universe nifty500          # download + cache daily candles
python -m indian_stocks scan     --universe nifty500          # today's watchlist + entry limits
python -m indian_stocks backtest --universe nifty500 --start 2022-01-01 --out results

python -m pytest                                              # 79 tests, no network needed
```

## The strategy, as implemented

**Stock selection** (`selection.py`, port of `_refresh_stocks`) — a stock is on the watchlist only if it passes, in order:

| # | Gate | Value (stocks book) |
|---|---|---|
| 1 | Volume floor — traded value of the last completed day | ≥ ₹4.5 crore (= the original's $500k × ₹90) |
| 2 | ADR floor — mean(high/low)−1 over 14 completed daily candles | ≥ 4 % |
| 3 | Relative strength — 30-day return minus Nifty 50's | ≥ +10 pts → **LONG leader** (≤ −10 pts → SHORT laggard, only if shorts are on). RS also **locks the stock's side** for every later entry |
| 4 | Daily ribbon — live daily EMA9/21/50 monotonically stacked | either direction |
| 5 | Near-the-high — last close vs 30-day closing high | within 10 % |
| 6 | Recent trend — A: 10-day return beats the benchmark's · B: EMA21 up ≥ 1 % over 5 days | |
| 7 | Lower-high — highest close of days 1–5 ago not below that of days 6–10 ago | |

Open positions keep being managed after a stock drops off the watchlist; it just takes no new entries.
Optional benchmark regime gate (Nifty 50 vs its EMA21), off by default as in the original.

**Entry** (`entry.py`, port of `_select_setup` / `_maintain_entry_limit`) — a **limit on the live EMA9** is rested only if:
live EMA9 ≥ EMA21 ≥ EMA50 · RS side agrees · daily ribbon agrees · the last completed close is above EMA9/21/50 ·
near-high ≤ 10 % · the 30-day closing high is ≤ 25 % above the limit · no lower high · the live price is still
*above* the limit (otherwise it would fill at market — "don't chase"). One entry per stock per day.
Size ₹9 lakh long (= $10,000) / ₹2.25 lakh short (= $2,500), whole shares.

**Exits** (evaluated once per completed daily candle, executed at the next open):
1. **Lower-close rule on EMA9** — in profit *or* loss: once closes are below EMA9, exit on the first close
   *lower* than the previous one while still below; a close back above EMA9 cancels; 3 closes below with no
   lower close also exits (`_weak_run_decision`).
2. **Ribbon flip** — EMA9 ≤ EMA21 (`exit_mode = ema9_ema21`; `ema9_ema50` and the full ribbon are options).

**Stops & take-profits**
* Hard stop **−10 %** from the fill (a gap through it fills at the open).
* **TP1**: sell 30 % at +20 % (resting limit, re-sized after adds) → the runner's stop moves to **breakeven**.
* **TP2** (after TP1): sell 50 % of the remainder at `daily EMA50 × (1 + 7 × ADR)`, refreshed daily.
* A stop blocks re-entry for the rest of that day.

**Scale-in**: from the next day, one add per day, only when the position is **≥ +20 %** in profit, limit on EMA9
(same ribbon / daily-ribbon / RS gates), sized 50 % → 40 % → 30 % → 20 % → 10 % (floor) of the initial position,
max 10 adds. After an add the tiers collapse to **one stop 3 % under the new average**.

## Deviations from the original (read these)

| Original (Hyperliquid HIP-3 stock perps) | Here | Why |
|---|---|---|
| USD sizes and USD volume floor | INR at `USD_INR = 90` in `config.py` | NSE is INR |
| 5× isolated leverage | `leverage = 1.0` (a knob) | NSE cash delivery. Margin/F&O is not modelled |
| `ENABLE_SHORTS = False` | same — **off** | You cannot hold a short overnight in the NSE cash segment (F&O only). The mirror logic is implemented and tested: `--shorts true` |
| S&P 500 benchmark | Nifty 50 (`^NSEI` on Yahoo, `NIFTYBEES` for `nse`/`csv`) | |
| 24/7 UTC candles | NSE session candles (09:15–15:30 IST); N-hour bars never straddle sessions | |
| Fractional size | whole shares, prices on the ₹0.05 tick | |
| Exchange fees | flat 12 bps/side (brokerage+STT+levies, rough) + 5 bps slippage on market-type fills | **assumption — set your own** (`ExecConfig`) |
| Shipped `signal_interval = "1d"` (the file/branch name says 1h) | **1d** default; `--hourly` + `--signal-interval 4h` etc. available | follows the shipped config |
| Each symbol independent in the old backtester | one shared account: entries/adds need free margin | more realistic |

Also: the original backtester in `clude_crypto/backtest` has drifted from the live bot (it models a different
"N-close" exit, `LOWER_HIGH_PRIOR_DAYS = 10`, `EMA50_TP_MULT = 5`). This repo follows the **live bot**
(lower-close rule, 5, 7).

## How the simulation works — and its limits

The live bot polls every 30 minutes. The engine steps over bars (daily; hourly with `--hourly`) and at each
bar's **open** runs one bot cycle with live price = open, using only completed candles plus that open. Then the
bar's high/low is replayed against resting orders, **adverse first**: stops → entry fill → add fill → take-profits.

* **Daily mode has one decision point per day** instead of ~13 polls, and resting limits are priced once at the open.
* If one bar both fills an entry and touches its stop, **the stop is taken** (the intrabar order is unknowable).
* EMAs use full-history recursion, the bot a 250-candle window — equal after warm-up (verified to 1e-9 on the same input).
* Not modelled: circuit limits, T2T/ASM-GSM restrictions, dividends (use adjusted data), partial fills, order-book depth.
* **ADR ≥ 4 % is a high bar** for Indian stocks (large caps typically run well under that), so expect the watchlist to
  skew to volatile mid/small caps and to be short in quiet markets. Lower it with `--adr-min` if you want large caps.
* Results on a filtered trend-following selection are very sensitive to the symbol list (survivorship) and costs.
  Treat them as relative, not predictive.

## Data

| `--provider` | Source | Notes |
|---|---|---|
| `yahoo` (default) | `yfinance`, `<SYMBOL>.NS`, `^NSEI` | split/dividend **adjusted**; daily history is long, hourly only ~730 days; turnover ≈ close × volume |
| `nse` | NSE full bhavcopy `sec_bhavdata_full_DDMMYYYY.csv` | official, exact traded value, daily only, **unadjusted** (splits/bonuses create fake gaps) |
| `csv` | your own `<SYMBOL>.csv` (`date,open,high,low,close,volume`) | offline / bring-your-own data |

The stock list comes from NSE's index CSVs (`nifty50/100/200/500`) or `EQUITY_L.csv` (`all`). Only ordinary
EQ-series shares are kept (ETFs, index funds, InvITs, REITs, T2T/SME series are dropped — heuristic, use
`--blacklist` for strays). If NSE is unreachable, a short bundled starter list (`data/universe_fallback.csv`,
31 large caps — **not** real index membership) is used and a warning is printed.

> **Verification status.** The Yahoo and NSE download paths were written against their documented formats and
> tested with mocked responses only: the sandbox this was built in blocks `nseindia.com` and Yahoo. Run
> `python -m indian_stocks fetch --max-symbols 5` once with internet access to confirm them before relying on
> any backtest. Everything else (rules, selection, engine, accounting, CLI) is tested offline.

## Tests

`python -m pytest` — 79 tests:

* **Parity with the original** (`tests/test_parity.py`): the original bot's `calculate_ema`, `_weak_run_decision`,
  `_raw_setup`, `_closed_candle_confirms`, `_daily_ribbon_confirms` and the whole `_check_exit` ladder are
  extracted from its source with `ast` and run next to the ports on thousands of random inputs. Set `EMA_BOT_PATH`
  to the original file; the module is skipped if it is absent.
* Selection gates, engine scenarios on hand-built tapes (limit fills, gap stops, TP1 + breakeven, lower-close
  exit, add gating and stop re-basing, margin rejection), accounting identity, data parsing, CLI end-to-end.

## What is not here

* **Broker integration / live order placement** (Zerodha Kite, Upstox, Dhan, …) — needs your API credentials
  and decisions about order types, product (CNC/MIS/NRML) and risk limits.
* Telegram alerts, trade logging, restart-sync and the rate-limit handling of the original bot.
* A short-selling model (F&O lots, margins, rollover).
