"""
Configuration for the Quant Trading Bot.

Copy this file to .env or set environment variables directly.
Required environment variables:
  ROOSTOO_API_KEY    - Your Roostoo API key
  ROOSTOO_SECRET_KEY - Your Roostoo secret key
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Credentials ────────────────────────────────────────────────────────────
API_KEY: str = os.environ.get("ROOSTOO_API_KEY", "")
SECRET_KEY: str = os.environ.get("ROOSTOO_SECRET_KEY", "")

BASE_URL: str = "https://mock-api.roostoo.com"

# ── Trading Universe ───────────────────────────────────────────────────────────
# Pairs to trade; the bot discovers all available pairs at startup and falls
# back to this list if the exchange info endpoint is unavailable.
DEFAULT_PAIRS: list[str] = [
    "BTC/USD",
    "ETH/USD",
    "BNB/USD",
    "EOS/USD",
    "ETC/USD",
]

# ── Strategy Parameters ────────────────────────────────────────────────────────
# Number of price ticks to keep in the rolling window
PRICE_WINDOW: int = 60          # ~1 hour of 1-min data

# RSI settings
RSI_PERIOD: int = 14
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0

# MACD settings (EMA periods)
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# Bollinger Bands settings
BB_PERIOD: int = 20
BB_STD_DEV: float = 2.0

# EMA trend filter
EMA_FAST: int = 9
EMA_SLOW: int = 21

# Minimum number of data points required before generating signals
MIN_DATA_POINTS: int = max(MACD_SLOW + MACD_SIGNAL, BB_PERIOD, RSI_PERIOD) + 5

# ── Risk Management ────────────────────────────────────────────────────────────
# Maximum fraction of portfolio allocated to a single asset
MAX_POSITION_FRACTION: float = 0.20   # 20 %

# Fixed fractional size of each trade as a fraction of available USD
TRADE_FRACTION: float = 0.10          # 10 % of free USD per trade

# Stop-loss / take-profit thresholds (relative to entry price)
STOP_LOSS_PCT: float = 0.05           # −5 %
TAKE_PROFIT_PCT: float = 0.10         # +10 %

# Maximum portfolio drawdown before the bot pauses new trades
MAX_DRAWDOWN_LIMIT: float = 0.20      # 20 %

# ── Bot Loop ───────────────────────────────────────────────────────────────────
# Seconds to sleep between each strategy iteration
LOOP_INTERVAL_SECONDS: int = 60

# Risk-free rate for Sharpe / Sortino (annualised, decimal form)
RISK_FREE_RATE: float = 0.02          # 2 % p.a.

# Number of trading periods per year used for ratio annualisation
# (assuming 1-minute bars, crypto trades 24/7 ≈ 525 600 min/year)
PERIODS_PER_YEAR: int = 525_600
