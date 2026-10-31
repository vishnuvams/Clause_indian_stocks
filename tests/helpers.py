"""Synthetic candle builders for tests (no network, deterministic)."""
import numpy as np
import pandas as pd


def bdays(n, start="2021-01-04"):
    return pd.bdate_range(start, periods=n)


def make_df(opens, highs, lows, closes, volume=2_000_000, start="2021-01-04"):
    n = len(closes)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                       "volume": [volume] * n}, index=bdays(n, start))
    df["turnover"] = df["close"] * df["volume"]
    return df


def trend(n, p0=100.0, g=0.01, spread=0.025, volume=2_000_000, start="2021-01-04"):
    """Steady compounding uptrend; each bar opens at the prior close; +/- `spread` high/low."""
    closes = [p0 * (1 + g) ** (i + 1) for i in range(n)]
    opens = [p0] + closes[:-1]
    highs = [max(o, c) * (1 + spread) for o, c in zip(opens, closes)]
    lows = [min(o, c) * (1 - spread) for o, c in zip(opens, closes)]
    return make_df(opens, highs, lows, closes, volume, start)


def flat(n, p=100.0, spread=0.004, volume=50_000_000, start="2021-01-04"):
    return make_df([p] * n, [p * (1 + spread)] * n, [p * (1 - spread)] * n, [p] * n, volume, start)


def random_walk(n, seed, p0=100.0, drift=0.0005, vol=0.02, volume=3_000_000, start="2021-01-04"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    closes = p0 * np.cumprod(1 + rets)
    opens = np.concatenate([[p0], closes[:-1]]) * (1 + rng.normal(0, vol / 4, n))
    hi = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return make_df(opens.tolist(), hi.tolist(), lo.tolist(), closes.tolist(), volume, start)
