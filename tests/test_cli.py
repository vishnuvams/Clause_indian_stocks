import numpy as np
import pandas as pd

from indian_stocks import cli
from tests.helpers import flat, make_df


def _write_csv_book(folder, end, n=500):
    start = str(pd.bdate_range(end=end, periods=n)[0].date())
    g = 0.008 + 0.012 * np.sin(np.arange(n) / 9.0)
    closes = 100 * np.cumprod(1 + g)
    opens = np.concatenate([[100], closes[:-1]])
    hi, lo = np.maximum(opens, closes) * 1.025, np.minimum(opens, closes) * 0.975
    for name, scale in (("AAA", 1.0), ("BBB", 0.6)):
        df = make_df((opens * scale).tolist(), (hi * scale).tolist(), (lo * scale).tolist(),
                     (closes * scale).tolist(), volume=5_000_000, start=start)
        df[["open", "high", "low", "close", "volume"]].rename_axis("date").to_csv(folder / f"{name}.csv")
    flat(n, start=start)[["open", "high", "low", "close", "volume"]].rename_axis("date").to_csv(folder / "NIFTYBEES.csv")


def test_cli_scan_lists_the_watchlist(tmp_path, capsys):
    today = cli._ist_today()
    _write_csv_book(tmp_path, today - pd.Timedelta(days=1) if today.weekday() else today - pd.Timedelta(days=3))
    out = tmp_path / "scan.csv"
    cli.main(["scan", "--provider", "csv", "--data-dir", str(tmp_path), "--symbols", "AAA", "BBB", "--out", str(out)])
    assert "stocks on the watchlist" in capsys.readouterr().out
    df = pd.read_csv(out)
    assert set(df["symbol"]) == {"AAA", "BBB"} and {"selected", "why", "entry_limit", "stop", "tp1"} <= set(df.columns)


def test_cli_universe_offline(capsys):
    cli.main(["universe", "--universe", "nifty50", "--offline", "--cache-dir", "/nonexistent-cache"])
    assert "RELIANCE" in capsys.readouterr().out


def test_cli_has_no_backtest_command():
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["backtest"])
