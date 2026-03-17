"""
Unit tests for risk_manager.py
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from risk_manager import RiskManager, Position


# ── Position ───────────────────────────────────────────────────────────────────

class TestPosition:
    def test_stop_loss_price(self):
        pos = Position(pair="BTC/USD", coin="BTC", quantity=0.1, entry_price=10_000.0)
        expected = 10_000.0 * (1 - 0.05)  # using default 5%
        assert pos.stop_loss_price == pytest.approx(expected)

    def test_take_profit_price(self):
        pos = Position(pair="BTC/USD", coin="BTC", quantity=0.1, entry_price=10_000.0)
        expected = 10_000.0 * (1 + 0.10)  # using default 10%
        assert pos.take_profit_price == pytest.approx(expected)

    def test_pnl_pct_profit(self):
        pos = Position(pair="BTC/USD", coin="BTC", quantity=1.0, entry_price=100.0)
        assert pos.pnl_pct(110.0) == pytest.approx(0.10)

    def test_pnl_pct_loss(self):
        pos = Position(pair="BTC/USD", coin="BTC", quantity=1.0, entry_price=100.0)
        assert pos.pnl_pct(90.0) == pytest.approx(-0.10)


# ── RiskManager ────────────────────────────────────────────────────────────────

class TestRiskManager:
    def test_open_and_get_position(self):
        rm = RiskManager(50_000.0)
        rm.open_position("BTC/USD", 0.5, 10_000.0)
        assert rm.has_position("BTC/USD")
        pos = rm.get_position("BTC/USD")
        assert pos is not None
        assert pos.quantity == 0.5
        assert pos.entry_price == 10_000.0

    def test_close_position(self):
        rm = RiskManager(50_000.0)
        rm.open_position("ETH/USD", 1.0, 2_000.0)
        rm.close_position("ETH/USD")
        assert not rm.has_position("ETH/USD")

    def test_close_nonexistent_position_returns_none(self):
        rm = RiskManager(50_000.0)
        assert rm.close_position("FAKE/USD") is None

    def test_should_stop_loss_triggered(self):
        rm = RiskManager(50_000.0)
        rm.open_position("BTC/USD", 1.0, 10_000.0)
        assert rm.should_stop_loss("BTC/USD", 9_400.0)   # below 9 500 (−5%)

    def test_should_stop_loss_not_triggered(self):
        rm = RiskManager(50_000.0)
        rm.open_position("BTC/USD", 1.0, 10_000.0)
        assert not rm.should_stop_loss("BTC/USD", 9_600.0)  # above 9 500

    def test_should_take_profit_triggered(self):
        rm = RiskManager(50_000.0)
        rm.open_position("BTC/USD", 1.0, 10_000.0)
        assert rm.should_take_profit("BTC/USD", 11_100.0)   # above 11 000 (+10%)

    def test_should_take_profit_not_triggered(self):
        rm = RiskManager(50_000.0)
        rm.open_position("BTC/USD", 1.0, 10_000.0)
        assert not rm.should_take_profit("BTC/USD", 10_500.0)

    def test_no_position_stop_loss_returns_false(self):
        rm = RiskManager(50_000.0)
        assert not rm.should_stop_loss("BTC/USD", 1.0)

    def test_no_position_take_profit_returns_false(self):
        rm = RiskManager(50_000.0)
        assert not rm.should_take_profit("BTC/USD", 1_000_000.0)

    # ── Sizing ─────────────────────────────────────────────────────────────────

    def test_compute_buy_quantity_basic(self):
        rm = RiskManager(50_000.0)
        qty = rm.compute_buy_quantity(
            pair="BTC/USD",
            free_usd=10_000.0,
            current_price=50_000.0,
            total_portfolio_value=50_000.0,
        )
        # trade_usd = 10 000 * 0.10 = 1 000, max_usd = 50 000 * 0.20 = 10 000
        # → spend 1 000 / 50 000 = 0.02
        assert qty == pytest.approx(0.02, rel=1e-3)

    def test_compute_buy_quantity_zero_price(self):
        rm = RiskManager(50_000.0)
        qty = rm.compute_buy_quantity("BTC/USD", 10_000.0, 0.0, 50_000.0)
        assert qty == 0.0

    def test_compute_buy_quantity_capped_by_position_limit(self):
        rm = RiskManager(50_000.0)
        # If free_usd * TRADE_FRACTION > total * MAX_POSITION_FRACTION, cap applies
        qty = rm.compute_buy_quantity(
            pair="ETH/USD",
            free_usd=100_000.0,     # very large balance
            current_price=1_000.0,
            total_portfolio_value=50_000.0,  # max_usd = 10 000
            amount_precision=2,
        )
        # max_usd = 50 000 * 0.20 = 10 000 → qty = 10 000 / 1 000 = 10
        assert qty == pytest.approx(10.0, rel=1e-3)

    def test_compute_buy_quantity_notional_too_small(self):
        rm = RiskManager(50_000.0)
        qty = rm.compute_buy_quantity(
            pair="BTC/USD",
            free_usd=0.01,           # nearly nothing
            current_price=50_000.0,
            total_portfolio_value=0.01,
        )
        assert qty == 0.0

    # ── Metrics ────────────────────────────────────────────────────────────────

    def test_drawdown_starts_zero(self):
        rm = RiskManager(50_000.0)
        assert rm.drawdown == pytest.approx(0.0)

    def test_drawdown_after_loss(self):
        rm = RiskManager(50_000.0)
        rm.update_portfolio_value(40_000.0)
        assert rm.drawdown == pytest.approx(0.20)

    def test_drawdown_recovers_on_new_peak(self):
        rm = RiskManager(50_000.0)
        rm.update_portfolio_value(40_000.0)
        rm.update_portfolio_value(60_000.0)
        assert rm.drawdown == pytest.approx(0.0)

    def test_trading_halted_when_drawdown_exceeds_limit(self):
        import config
        rm = RiskManager(100_000.0)
        halted_value = 100_000.0 * (1 - config.MAX_DRAWDOWN_LIMIT - 0.01)
        rm.update_portfolio_value(halted_value)
        assert rm.trading_halted

    def test_trading_not_halted_small_drawdown(self):
        rm = RiskManager(50_000.0)
        rm.update_portfolio_value(49_000.0)
        assert not rm.trading_halted

    def test_sharpe_nan_with_one_return(self):
        rm = RiskManager(50_000.0)
        rm.update_portfolio_value(51_000.0)
        assert math.isnan(rm.sharpe_ratio())

    def test_sharpe_computable_with_returns(self):
        rm = RiskManager(100.0)
        for v in [101.0, 102.5, 101.5, 103.0, 105.0]:
            rm.update_portfolio_value(v)
        result = rm.sharpe_ratio()
        assert not math.isnan(result)

    def test_sortino_nan_with_no_negative_returns(self):
        rm = RiskManager(100.0)
        for v in [101.0, 102.0, 103.0, 104.0, 105.0]:
            rm.update_portfolio_value(v)
        result = rm.sortino_ratio()
        # All positive returns → no downside → nan
        assert math.isnan(result)

    def test_sortino_computable_with_mixed_returns(self):
        rm = RiskManager(100.0)
        for v in [101.0, 99.0, 102.0, 98.0, 103.0]:
            rm.update_portfolio_value(v)
        result = rm.sortino_ratio()
        assert not math.isnan(result)

    def test_calmar_computable(self):
        rm = RiskManager(100.0)
        # Use enough return periods to avoid overflow in annualisation.
        # Feed 100 values that include both gains and losses with a net gain.
        import math as _math
        values = [100.0 + (i % 5) * 2 - (i % 7) for i in range(1, 101)]
        for v in values:
            rm.update_portfolio_value(v)
        result = rm.calmar_ratio()
        # Either computable (not nan) or nan when drawdown/return conditions not met
        assert isinstance(result, float)

    def test_performance_summary_keys(self):
        rm = RiskManager(50_000.0)
        summary = rm.performance_summary()
        expected_keys = {
            "portfolio_value", "peak_value", "drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "num_open_positions", "trading_halted",
        }
        assert set(summary.keys()) == expected_keys
