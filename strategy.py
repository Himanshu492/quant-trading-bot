"""
Trading strategy module.

Implements a multi-signal hybrid strategy that combines:
  - RSI (momentum / mean-reversion)
  - MACD (trend-following)
  - Bollinger Bands (volatility / mean-reversion)
  - EMA crossover (trend filter)

Each indicator casts a vote (+1 = bullish, −1 = bearish, 0 = neutral) and the
votes are summed to produce a composite signal score.  The bot acts on the
signal when the score exceeds configurable thresholds.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional

import config
import indicators as ind

logger = logging.getLogger(__name__)


# ── Signal enum ────────────────────────────────────────────────────────────────


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ── Per-pair state ─────────────────────────────────────────────────────────────


@dataclass
class PairState:
    """Rolling price history and the latest computed signal for one pair."""

    pair: str
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=config.PRICE_WINDOW))
    last_signal: Signal = Signal.HOLD
    signal_score: float = 0.0

    def add_price(self, price: float) -> None:
        self.prices.append(price)

    @property
    def ready(self) -> bool:
        """True once enough price history is available to compute all indicators."""
        return len(self.prices) >= config.MIN_DATA_POINTS


# ── Strategy ───────────────────────────────────────────────────────────────────


class MultiSignalStrategy:
    """
    Compute BUY / SELL / HOLD signals for each tracked trading pair.

    Scoring system
    --------------
    Each indicator contributes a vote:
      RSI        : +1 (oversold), −1 (overbought), 0 (neutral)
      MACD       : +1 (histogram > 0 and rising), −1 (histogram < 0 and falling)
      Bollinger  : +1 (price near lower band), −1 (price near upper band)
      EMA cross  : +1 (fast EMA > slow EMA), −1 (fast < slow)

    Composite score ∈ [−4, +4].  Thresholds:
      score ≥ BUY_THRESHOLD  → BUY
      score ≤ SELL_THRESHOLD → SELL
      otherwise              → HOLD
    """

    BUY_THRESHOLD: float = 2.0
    SELL_THRESHOLD: float = -2.0

    def __init__(self) -> None:
        self._states: Dict[str, PairState] = {}

    # ── State management ───────────────────────────────────────────────────────

    def add_pair(self, pair: str) -> None:
        """Register a new trading pair to track."""
        if pair not in self._states:
            self._states[pair] = PairState(pair=pair)

    def update_price(self, pair: str, price: float) -> None:
        """Feed a new price tick into the state for *pair*."""
        if pair not in self._states:
            self.add_pair(pair)
        self._states[pair].add_price(price)

    def get_state(self, pair: str) -> Optional[PairState]:
        return self._states.get(pair)

    # ── Signal computation ─────────────────────────────────────────────────────

    def compute_signal(self, pair: str) -> Signal:
        """
        Compute and return the current trading signal for *pair*.

        Returns ``Signal.HOLD`` if there is insufficient price history.
        """
        state = self._states.get(pair)
        if state is None or not state.ready:
            return Signal.HOLD

        prices = list(state.prices)
        score = self._score(prices)
        state.signal_score = score

        if score >= self.BUY_THRESHOLD:
            state.last_signal = Signal.BUY
        elif score <= self.SELL_THRESHOLD:
            state.last_signal = Signal.SELL
        else:
            state.last_signal = Signal.HOLD

        logger.debug("Pair=%s score=%.1f signal=%s", pair, score, state.last_signal.value)
        return state.last_signal

    def _score(self, prices: list[float]) -> float:
        """Return the composite signal score for a price series."""
        score = 0.0

        # ── RSI vote ──────────────────────────────────────────────────────────
        try:
            rsi_val = ind.rsi(prices, config.RSI_PERIOD)
            if not math.isnan(rsi_val):
                if rsi_val < config.RSI_OVERSOLD:
                    score += 1.0
                elif rsi_val > config.RSI_OVERBOUGHT:
                    score -= 1.0
        except ValueError:
            pass

        # ── MACD vote ─────────────────────────────────────────────────────────
        try:
            _, _, histogram = ind.macd(
                prices, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
            )
            if histogram > 0:
                score += 1.0
            elif histogram < 0:
                score -= 1.0
        except ValueError:
            pass

        # ── Bollinger Bands vote ───────────────────────────────────────────────
        try:
            upper, middle, lower = ind.bollinger_bands(
                prices, config.BB_PERIOD, config.BB_STD_DEV
            )
            current_price = prices[-1]
            band_width = upper - lower
            if band_width > 0:
                position = (current_price - lower) / band_width  # 0 = at lower, 1 = at upper
                if position < 0.2:
                    score += 1.0  # near lower band → bullish
                elif position > 0.8:
                    score -= 1.0  # near upper band → bearish
        except ValueError:
            pass

        # ── EMA crossover vote ────────────────────────────────────────────────
        try:
            fast_ema = ind.ema(prices, config.EMA_FAST)
            slow_ema = ind.ema(prices, config.EMA_SLOW)
            if fast_ema > slow_ema:
                score += 1.0
            elif fast_ema < slow_ema:
                score -= 1.0
        except ValueError:
            pass

        return score

    # ── Convenience ────────────────────────────────────────────────────────────

    @property
    def pairs(self) -> list[str]:
        return list(self._states.keys())
