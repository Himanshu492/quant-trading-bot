"""
Risk management module.

Responsibilities
----------------
- Track open positions and their entry prices.
- Decide whether a stop-loss or take-profit has been triggered.
- Size each trade using a fixed-fractional approach capped by a
  per-asset maximum allocation.
- Compute performance metrics: Sharpe ratio, Sortino ratio, Calmar ratio.
- Pause trading if the portfolio drawdown exceeds the configured limit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


# ── Position tracking ──────────────────────────────────────────────────────────


@dataclass
class Position:
    """Represents a currently open position in a single coin."""

    pair: str
    coin: str
    quantity: float
    entry_price: float

    @property
    def stop_loss_price(self) -> float:
        return self.entry_price * (1.0 - config.STOP_LOSS_PCT)

    @property
    def take_profit_price(self) -> float:
        return self.entry_price * (1.0 + config.TAKE_PROFIT_PCT)

    def pnl_pct(self, current_price: float) -> float:
        """Unrealised P&L as a fraction of entry price."""
        return (current_price - self.entry_price) / self.entry_price


# ── Risk manager ───────────────────────────────────────────────────────────────


class RiskManager:
    """
    Central risk-management component.

    Parameters
    ----------
    initial_portfolio_value : float
        The starting portfolio value (USD) used to track peak and drawdown.
    """

    def __init__(self, initial_portfolio_value: float = 50_000.0) -> None:
        self._positions: Dict[str, Position] = {}      # keyed by pair
        self._portfolio_returns: List[float] = []       # period-over-period returns
        self._peak_value: float = initial_portfolio_value
        self._initial_value: float = initial_portfolio_value
        self._current_value: float = initial_portfolio_value

    # ── Position helpers ───────────────────────────────────────────────────────

    def open_position(
        self, pair: str, quantity: float, entry_price: float
    ) -> Position:
        """Record a new long position for *pair*."""
        coin = pair.split("/")[0]
        position = Position(
            pair=pair, coin=coin, quantity=quantity, entry_price=entry_price
        )
        self._positions[pair] = position
        logger.info(
            "Opened position: %s qty=%.6f @ %.4f", pair, quantity, entry_price
        )
        return position

    def close_position(self, pair: str) -> Optional[Position]:
        """Remove and return the open position for *pair*, or None."""
        pos = self._positions.pop(pair, None)
        if pos:
            logger.info("Closed position: %s", pair)
        return pos

    def get_position(self, pair: str) -> Optional[Position]:
        return self._positions.get(pair)

    def has_position(self, pair: str) -> bool:
        return pair in self._positions

    # ── Exit signal checks ─────────────────────────────────────────────────────

    def should_stop_loss(self, pair: str, current_price: float) -> bool:
        """Return True if the stop-loss threshold has been breached."""
        pos = self._positions.get(pair)
        if pos is None:
            return False
        triggered = current_price <= pos.stop_loss_price
        if triggered:
            logger.warning(
                "Stop-loss triggered for %s: entry=%.4f current=%.4f",
                pair, pos.entry_price, current_price,
            )
        return triggered

    def should_take_profit(self, pair: str, current_price: float) -> bool:
        """Return True if the take-profit threshold has been reached."""
        pos = self._positions.get(pair)
        if pos is None:
            return False
        triggered = current_price >= pos.take_profit_price
        if triggered:
            logger.info(
                "Take-profit triggered for %s: entry=%.4f current=%.4f",
                pair, pos.entry_price, current_price,
            )
        return triggered

    # ── Trade sizing ───────────────────────────────────────────────────────────

    def compute_buy_quantity(
        self,
        pair: str,
        free_usd: float,
        current_price: float,
        total_portfolio_value: float,
        amount_precision: int = 6,
    ) -> float:
        """
        Compute the quantity to buy for a market order.

        Uses a fixed-fractional approach:
          trade_usd = free_usd * TRADE_FRACTION

        Capped so the resulting position does not exceed
        MAX_POSITION_FRACTION of the total portfolio.

        Returns 0.0 when the calculated quantity would be below the exchange
        minimum of 1.0 USD notional.
        """
        trade_usd = free_usd * config.TRADE_FRACTION
        max_usd = total_portfolio_value * config.MAX_POSITION_FRACTION
        usd_to_spend = min(trade_usd, max_usd)

        if current_price <= 0:
            return 0.0

        quantity = usd_to_spend / current_price
        # Round to exchange precision
        factor = 10 ** amount_precision
        quantity = math.floor(quantity * factor) / factor

        # Enforce the exchange minimum order size (1 USD notional)
        if quantity * current_price < 1.0:
            return 0.0

        return quantity

    # ── Portfolio metrics ──────────────────────────────────────────────────────

    def update_portfolio_value(self, new_value: float) -> None:
        """Record the current portfolio value and compute the period return."""
        if self._current_value > 0:
            period_return = (new_value - self._current_value) / self._current_value
            self._portfolio_returns.append(period_return)
        self._current_value = new_value
        if new_value > self._peak_value:
            self._peak_value = new_value

    @property
    def drawdown(self) -> float:
        """Current drawdown from the all-time peak (as a positive fraction)."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    @property
    def trading_halted(self) -> bool:
        """True when the drawdown exceeds the configured limit."""
        return self.drawdown >= config.MAX_DRAWDOWN_LIMIT

    def sharpe_ratio(self, risk_free_rate: float = config.RISK_FREE_RATE) -> float:
        """
        Annualised Sharpe ratio.

        Returns ``math.nan`` when there are fewer than 2 data points or
        when the standard deviation of returns is zero.
        """
        returns = self._portfolio_returns
        if len(returns) < 2:
            return math.nan
        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        if std_r == 0:
            return math.nan
        period_rf = risk_free_rate / config.PERIODS_PER_YEAR
        excess = mean_r - period_rf
        return (excess / std_r) * math.sqrt(config.PERIODS_PER_YEAR)

    def sortino_ratio(self, risk_free_rate: float = config.RISK_FREE_RATE) -> float:
        """
        Annualised Sortino ratio (uses downside deviation).

        Returns ``math.nan`` when there are insufficient data points or
        no negative returns.
        """
        returns = self._portfolio_returns
        if len(returns) < 2:
            return math.nan
        n = len(returns)
        mean_r = sum(returns) / n
        period_rf = risk_free_rate / config.PERIODS_PER_YEAR
        downside_sq = [r ** 2 for r in returns if r < period_rf]
        if not downside_sq:
            return math.nan
        downside_dev = math.sqrt(sum(downside_sq) / n)
        if downside_dev == 0:
            return math.nan
        return ((mean_r - period_rf) / downside_dev) * math.sqrt(config.PERIODS_PER_YEAR)

    def calmar_ratio(self) -> float:
        """
        Calmar ratio: annualised return / maximum drawdown.

        Returns ``math.nan`` when there are insufficient data, zero drawdown,
        or when the annualisation exponent would overflow (too few periods).
        """
        returns = self._portfolio_returns
        if len(returns) < 2 or self._initial_value <= 0:
            return math.nan
        total_return = (self._current_value - self._initial_value) / self._initial_value
        periods = len(returns)
        exponent = config.PERIODS_PER_YEAR / periods
        # Guard against overflow when period count is much smaller than PERIODS_PER_YEAR
        try:
            annualised_return = (1 + total_return) ** exponent - 1
        except OverflowError:
            return math.nan
        max_dd = self._max_drawdown()
        if max_dd <= 0:
            return math.nan
        return annualised_return / max_dd

    def _max_drawdown(self) -> float:
        """Compute the maximum drawdown experienced over all recorded returns."""
        if not self._portfolio_returns:
            return 0.0
        peak = self._initial_value
        current = self._initial_value
        max_dd = 0.0
        for r in self._portfolio_returns:
            current *= 1 + r
            if current > peak:
                peak = current
            if peak > 0:
                dd = (peak - current) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def performance_summary(self) -> dict:
        """Return a dict with key risk/return metrics."""
        return {
            "portfolio_value": round(self._current_value, 4),
            "peak_value": round(self._peak_value, 4),
            "drawdown_pct": round(self.drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 4) if not math.isnan(self.sharpe_ratio()) else None,
            "sortino_ratio": round(self.sortino_ratio(), 4) if not math.isnan(self.sortino_ratio()) else None,
            "calmar_ratio": round(self.calmar_ratio(), 4) if not math.isnan(self.calmar_ratio()) else None,
            "num_open_positions": len(self._positions),
            "trading_halted": self.trading_halted,
        }
