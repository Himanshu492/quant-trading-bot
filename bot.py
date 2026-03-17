"""
Main trading bot loop.

Run with:
    python bot.py

Environment variables (or .env file):
    ROOSTOO_API_KEY    – Your Roostoo API key
    ROOSTOO_SECRET_KEY – Your Roostoo secret key
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import config
from risk_manager import RiskManager
from roostoo_client import RoostooClient
from strategy import MultiSignalStrategy, Signal

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bot")


# ── Bot class ──────────────────────────────────────────────────────────────────


class TradingBot:
    """
    Autonomous trading bot for the Roostoo mock exchange.

    Each iteration of the main loop:
      1. Fetch ticker prices for all tracked pairs.
      2. Feed prices into the multi-signal strategy.
      3. Evaluate stop-loss / take-profit for open positions.
      4. Generate and (optionally) execute trade orders.
      5. Update portfolio metrics.
    """

    def __init__(
        self,
        client: Optional[RoostooClient] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        client:
            Roostoo API client.  Created automatically when not supplied.
        dry_run:
            When *True*, the bot computes signals but does **not** place real
            orders on the exchange.  Useful for back-testing or debugging.
        """
        self.client = client or RoostooClient()
        self.strategy = MultiSignalStrategy()
        self.risk_manager: Optional[RiskManager] = None
        self.dry_run = dry_run
        self._pairs: list[str] = []
        self._pair_info: dict = {}   # precision etc. from exchangeInfo

    # ── Startup ────────────────────────────────────────────────────────────────

    def initialise(self) -> bool:
        """
        Connect to the exchange, discover trading pairs, and seed the risk
        manager with the current portfolio value.

        Returns True on success, False on failure.
        """
        logger.info("Connecting to Roostoo exchange …")
        server_time = self.client.get_server_time()
        if server_time is None:
            logger.error("Cannot reach exchange. Aborting.")
            return False
        logger.info("Server time: %s", server_time)

        info = self.client.get_exchange_info()
        if info and info.get("TradePairs"):
            self._pair_info = info["TradePairs"]
            self._pairs = [
                pair
                for pair, details in self._pair_info.items()
                if details.get("CanTrade", False)
            ]
            logger.info("Tradable pairs: %s", self._pairs)
        else:
            logger.warning(
                "Could not fetch exchange info; falling back to default pairs."
            )
            self._pairs = config.DEFAULT_PAIRS

        for pair in self._pairs:
            self.strategy.add_pair(pair)

        # Seed risk manager with current portfolio value
        initial_value = self._get_portfolio_value()
        self.risk_manager = RiskManager(initial_portfolio_value=initial_value)
        logger.info("Initial portfolio value: %.2f USD", initial_value)
        return True

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the trading loop indefinitely (until interrupted)."""
        if not self.initialise():
            return

        logger.info(
            "Bot started. dry_run=%s interval=%ss",
            self.dry_run, config.LOOP_INTERVAL_SECONDS,
        )

        iteration = 0
        while True:
            try:
                self._iterate(iteration)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down …")
                break
            except Exception as exc:
                logger.exception("Unexpected error in trading loop: %s", exc)

            iteration += 1
            time.sleep(config.LOOP_INTERVAL_SECONDS)

    def _iterate(self, iteration: int) -> None:
        """Execute one full strategy iteration."""
        # ── 1. Fetch prices ────────────────────────────────────────────────────
        ticker_response = self.client.get_ticker()
        if ticker_response is None or not ticker_response.get("Success"):
            logger.warning("Failed to fetch ticker data; skipping iteration.")
            return

        ticker_data: dict = ticker_response.get("Data", {})

        # ── 2. Update strategy with new prices ────────────────────────────────
        for pair in self._pairs:
            tick = ticker_data.get(pair)
            if tick is None:
                continue
            last_price: float = tick.get("LastPrice", 0.0)
            if last_price <= 0:
                continue
            self.strategy.update_price(pair, last_price)

        # ── 3. Risk checks on open positions ──────────────────────────────────
        assert self.risk_manager is not None
        for pair in list(self._pairs):
            if not self.risk_manager.has_position(pair):
                continue
            tick = ticker_data.get(pair)
            if tick is None:
                continue
            current_price: float = tick.get("LastPrice", 0.0)

            if self.risk_manager.should_stop_loss(pair, current_price):
                self._execute_sell(pair, current_price, reason="stop-loss")
            elif self.risk_manager.should_take_profit(pair, current_price):
                self._execute_sell(pair, current_price, reason="take-profit")

        # ── 4. Generate signals and execute ───────────────────────────────────
        if self.risk_manager.trading_halted:
            logger.warning(
                "Trading HALTED – drawdown %.1f%% exceeds limit %.1f%%.",
                self.risk_manager.drawdown * 100,
                config.MAX_DRAWDOWN_LIMIT * 100,
            )
        else:
            for pair in self._pairs:
                tick = ticker_data.get(pair)
                if tick is None:
                    continue
                current_price = tick.get("LastPrice", 0.0)
                signal = self.strategy.compute_signal(pair)
                state = self.strategy.get_state(pair)
                score = state.signal_score if state else 0.0
                logger.debug(
                    "Pair=%s price=%.4f signal=%s score=%.1f",
                    pair, current_price, signal.value, score,
                )

                if signal == Signal.BUY and not self.risk_manager.has_position(pair):
                    self._execute_buy(pair, current_price, ticker_data)
                elif signal == Signal.SELL and self.risk_manager.has_position(pair):
                    self._execute_sell(pair, current_price, reason="signal")

        # ── 5. Update portfolio metrics ────────────────────────────────────────
        new_value = self._get_portfolio_value(ticker_data)
        self.risk_manager.update_portfolio_value(new_value)

        if iteration % 10 == 0:
            summary = self.risk_manager.performance_summary()
            logger.info("Performance: %s", summary)

    # ── Trade execution ────────────────────────────────────────────────────────

    def _execute_buy(
        self,
        pair: str,
        current_price: float,
        ticker_data: dict,
    ) -> None:
        """Size and place a market BUY order."""
        balance = self.client.get_balance()
        if balance is None:
            return

        free_usd: float = balance.get("Wallet", {}).get("USD", {}).get("Free", 0.0)
        total_value = self._get_portfolio_value(ticker_data, balance)

        pair_info = self._pair_info.get(pair, {})
        amount_precision = pair_info.get("AmountPrecision", 6)

        assert self.risk_manager is not None
        quantity = self.risk_manager.compute_buy_quantity(
            pair=pair,
            free_usd=free_usd,
            current_price=current_price,
            total_portfolio_value=total_value,
            amount_precision=amount_precision,
        )
        if quantity <= 0:
            logger.info("BUY %s: quantity too small, skipping.", pair)
            return

        logger.info(
            "BUY  %s qty=%.6f @ ~%.4f%s",
            pair, quantity, current_price, " [DRY RUN]" if self.dry_run else "",
        )

        if not self.dry_run:
            result = self.client.place_order(pair, "BUY", quantity, order_type="MARKET")
            if result and result.get("Success"):
                filled_price = (
                    result["OrderDetail"].get("FilledAverPrice") or current_price
                )
                self.risk_manager.open_position(pair, quantity, filled_price)
            else:
                logger.warning("BUY order for %s failed: %s", pair, result)
        else:
            # In dry-run mode, record the position at the current market price
            self.risk_manager.open_position(pair, quantity, current_price)

    def _execute_sell(
        self,
        pair: str,
        current_price: float,
        reason: str = "signal",
    ) -> None:
        """Place a market SELL order to close the open position for *pair*."""
        assert self.risk_manager is not None
        position = self.risk_manager.get_position(pair)
        if position is None:
            return

        quantity = position.quantity
        logger.info(
            "SELL %s qty=%.6f @ ~%.4f (reason=%s)%s",
            pair, quantity, current_price, reason,
            " [DRY RUN]" if self.dry_run else "",
        )

        if not self.dry_run:
            result = self.client.place_order(pair, "SELL", quantity, order_type="MARKET")
            if result and result.get("Success"):
                self.risk_manager.close_position(pair)
            else:
                logger.warning("SELL order for %s failed: %s", pair, result)
        else:
            self.risk_manager.close_position(pair)

    # ── Portfolio valuation ────────────────────────────────────────────────────

    def _get_portfolio_value(
        self,
        ticker_data: Optional[dict] = None,
        balance: Optional[dict] = None,
    ) -> float:
        """
        Estimate total portfolio value in USD.

        Fetches the balance from the API if not supplied.  Falls back to the
        initial value when the API is unavailable.
        """
        if balance is None:
            balance = self.client.get_balance()
        if balance is None:
            return (
                self.risk_manager._current_value
                if self.risk_manager
                else 50_000.0
            )

        wallet: dict = balance.get("Wallet", {})
        usd_free: float = wallet.get("USD", {}).get("Free", 0.0)
        usd_lock: float = wallet.get("USD", {}).get("Lock", 0.0)
        total = usd_free + usd_lock

        # Add the current market value of all coin holdings
        for asset, amounts in wallet.items():
            if asset == "USD":
                continue
            coin_free: float = amounts.get("Free", 0.0)
            coin_lock: float = amounts.get("Lock", 0.0)
            coin_total = coin_free + coin_lock
            if coin_total <= 0:
                continue

            # Look for the price in ticker_data first; otherwise query the API
            pair = f"{asset}/USD"
            price: Optional[float] = None
            if ticker_data and pair in ticker_data:
                price = ticker_data[pair].get("LastPrice")
            if price is None:
                tick_resp = self.client.get_ticker(pair)
                if tick_resp and tick_resp.get("Success") and tick_resp.get("Data"):
                    price = tick_resp["Data"].get(pair, {}).get("LastPrice")
            if price and price > 0:
                total += coin_total * price

        return total


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    if not config.API_KEY or not config.SECRET_KEY:
        logger.error(
            "ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY must be set. "
            "Create a .env file or export the variables before running."
        )
        sys.exit(1)

    bot = TradingBot(dry_run=False)
    bot.run()


if __name__ == "__main__":
    main()
