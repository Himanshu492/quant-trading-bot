"""
Technical indicators used by the trading strategy.

All functions operate on plain Python lists or numpy arrays of prices and
return scalar values (or tuples of scalars).  They are intentionally
dependency-light so they can run without a full pandas installation.
"""

from __future__ import annotations

import math
from typing import Sequence


# ── Helpers ────────────────────────────────────────────────────────────────────


def _require(prices: Sequence[float], minimum: int, name: str) -> None:
    if len(prices) < minimum:
        raise ValueError(
            f"{name} requires at least {minimum} data points; "
            f"got {len(prices)}."
        )


# ── Simple / Exponential Moving Averages ──────────────────────────────────────


def sma(prices: Sequence[float], period: int) -> float:
    """Simple moving average of the last *period* values."""
    _require(prices, period, "SMA")
    window = list(prices)[-period:]
    return sum(window) / period


def ema(prices: Sequence[float], period: int) -> float:
    """
    Exponential moving average of *prices* with span *period*.

    Uses the standard smoothing factor k = 2 / (period + 1) and seeds the
    EMA with the SMA of the first *period* values.
    """
    _require(prices, period, "EMA")
    k = 2.0 / (period + 1)
    data = list(prices)
    result = sum(data[:period]) / period  # seed with SMA
    for price in data[period:]:
        result = price * k + result * (1 - k)
    return result


def ema_series(prices: Sequence[float], period: int) -> list[float]:
    """Return the full EMA series (same length as *prices*, NaN-padded)."""
    _require(prices, period, "EMA series")
    k = 2.0 / (period + 1)
    data = list(prices)
    result: list[float] = [math.nan] * (period - 1)
    current = sum(data[:period]) / period
    result.append(current)
    for price in data[period:]:
        current = price * k + current * (1 - k)
        result.append(current)
    return result


# ── Relative Strength Index ────────────────────────────────────────────────────


def rsi(prices: Sequence[float], period: int = 14) -> float:
    """
    Wilder's RSI.

    Returns a value in [0, 100].  A return value of ``math.nan`` is possible
    when the average loss over the window is zero (price only went up).
    """
    _require(prices, period + 1, "RSI")
    data = list(prices)[-(period + 1):]
    gains = []
    losses = []
    for i in range(1, len(data)):
        change = data[i] - data[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ── MACD ──────────────────────────────────────────────────────────────────────


def macd(
    prices: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """
    Moving Average Convergence/Divergence.

    Returns
    -------
    macd_line : float
        EMA(fast) − EMA(slow)
    signal_line : float
        EMA of the MACD line over the last *signal* periods.
    histogram : float
        macd_line − signal_line
    """
    _require(prices, slow + signal, "MACD")
    data = list(prices)

    # Build the full MACD-line series so we can compute EMA of it
    fast_series = ema_series(data, fast)
    slow_series = ema_series(data, slow)

    macd_line_series = [
        (f - s)
        for f, s in zip(fast_series, slow_series)
        if not (math.isnan(f) or math.isnan(s))
    ]
    _require(macd_line_series, signal, "MACD signal")

    signal_line = ema(macd_line_series, signal)
    current_macd = macd_line_series[-1]
    histogram = current_macd - signal_line
    return current_macd, signal_line, histogram


# ── Bollinger Bands ────────────────────────────────────────────────────────────


def bollinger_bands(
    prices: Sequence[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[float, float, float]:
    """
    Bollinger Bands.

    Returns
    -------
    upper : float
    middle : float   (SMA)
    lower : float
    """
    _require(prices, period, "Bollinger Bands")
    window = list(prices)[-period:]
    middle = sum(window) / period
    variance = sum((p - middle) ** 2 for p in window) / period
    std = math.sqrt(variance)
    return middle + std_dev * std, middle, middle - std_dev * std


# ── Average True Range (volatility) ───────────────────────────────────────────


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """
    Average True Range.

    *highs*, *lows*, *closes* must all have the same length ≥ *period* + 1.
    """
    n = len(closes)
    _require(closes, period + 1, "ATR")
    true_ranges: list[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period
