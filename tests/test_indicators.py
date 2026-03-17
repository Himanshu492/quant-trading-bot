"""
Unit tests for indicators.py
"""

import math
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import sma, ema, ema_series, rsi, macd, bollinger_bands, atr


# ── Helpers ────────────────────────────────────────────────────────────────────

def _linear(start: float, step: float, n: int) -> list[float]:
    """Return [start, start+step, …] of length n."""
    return [start + i * step for i in range(n)]


# ── SMA ────────────────────────────────────────────────────────────────────────

class TestSMA:
    def test_constant_series(self):
        prices = [10.0] * 20
        assert sma(prices, 10) == pytest.approx(10.0)

    def test_rising_series(self):
        # SMA of [1, 2, 3, 4, 5] = 3
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sma(prices, 5) == pytest.approx(3.0)

    def test_uses_last_period(self):
        prices = [100.0] * 10 + [200.0] * 5
        assert sma(prices, 5) == pytest.approx(200.0)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            sma([1.0, 2.0], 5)


# ── EMA ────────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_constant_series_equals_value(self):
        prices = [50.0] * 30
        assert ema(prices, 10) == pytest.approx(50.0)

    def test_returns_scalar(self):
        prices = list(range(1, 31))
        result = ema(prices, 10)
        assert isinstance(result, float)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            ema([1.0], 5)

    def test_ema_series_length(self):
        prices = list(range(1, 31))  # 30 values
        series = ema_series(prices, 5)
        assert len(series) == 30

    def test_ema_series_nan_prefix(self):
        prices = list(range(1, 11))   # 10 values
        series = ema_series(prices, 5)
        # First 4 values should be nan, 5th onwards should be finite
        for v in series[:4]:
            assert math.isnan(v)
        for v in series[4:]:
            assert not math.isnan(v)

    def test_ema_series_constant(self):
        prices = [7.0] * 20
        series = ema_series(prices, 5)
        for v in series[4:]:
            assert v == pytest.approx(7.0)


# ── RSI ────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_only_gains_returns_100(self):
        prices = _linear(10.0, 1.0, 20)
        assert rsi(prices, 14) == pytest.approx(100.0)

    def test_only_losses_returns_0(self):
        prices = _linear(200.0, -1.0, 20)
        result = rsi(prices, 14)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_mixed_returns_value_in_range(self):
        prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
            43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33,
        ]
        result = rsi(prices, 14)
        assert 0.0 <= result <= 100.0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            rsi([1.0, 2.0, 3.0], 14)


# ── MACD ──────────────────────────────────────────────────────────────────────

class TestMACD:
    def _long_prices(self):
        """Generate enough data for default MACD (26 + 9 = 35 min data points)."""
        return _linear(100.0, 0.5, 60)

    def test_returns_three_floats(self):
        prices = self._long_prices()
        result = macd(prices)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_histogram_equals_macd_minus_signal(self):
        prices = self._long_prices()
        macd_line, signal_line, histogram = macd(prices)
        assert histogram == pytest.approx(macd_line - signal_line, rel=1e-6)

    def test_trending_up_positive_macd(self):
        prices = _linear(100.0, 2.0, 60)
        macd_line, signal_line, _ = macd(prices)
        # In a strong uptrend EMA(fast) > EMA(slow) → macd > 0
        assert macd_line > 0

    def test_trending_down_negative_macd(self):
        prices = _linear(200.0, -2.0, 60)
        macd_line, _, _ = macd(prices)
        assert macd_line < 0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            macd([1.0, 2.0, 3.0])


# ── Bollinger Bands ────────────────────────────────────────────────────────────

class TestBollingerBands:
    def test_constant_series_zero_width(self):
        prices = [100.0] * 25
        upper, middle, lower = bollinger_bands(prices, period=20)
        assert upper == pytest.approx(100.0)
        assert middle == pytest.approx(100.0)
        assert lower == pytest.approx(100.0)

    def test_upper_gt_middle_gt_lower(self):
        import random
        random.seed(42)
        prices = [100.0 + random.gauss(0, 5) for _ in range(30)]
        upper, middle, lower = bollinger_bands(prices)
        assert upper >= middle >= lower

    def test_middle_equals_sma(self):
        prices = list(range(1, 26))
        _, middle, _ = bollinger_bands(prices, period=20)
        expected_sma = sma(prices, 20)
        assert middle == pytest.approx(expected_sma)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            bollinger_bands([1.0, 2.0], period=20)


# ── ATR ────────────────────────────────────────────────────────────────────────

class TestATR:
    def test_zero_range_returns_zero(self):
        prices = [100.0] * 20
        result = atr(prices, prices, prices, period=14)
        assert result == pytest.approx(0.0)

    def test_returns_positive_value(self):
        highs = _linear(110.0, 0.5, 20)
        lows = _linear(90.0, 0.5, 20)
        closes = _linear(100.0, 0.5, 20)
        result = atr(highs, lows, closes, period=14)
        assert result > 0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            atr([1.0], [0.9], [1.0], period=14)
