# quant-trading-bot

An autonomous AI/quant trading bot for the [Roostoo mock exchange](https://github.com/roostoo/Roostoo-API-Documents).

The bot makes **buy / sell / hold** decisions without manual intervention by combining four technical indicators into a composite signal score, then enforces robust risk controls before placing orders.

---

## Architecture

```
quant-trading-bot/
├── bot.py            # Main trading loop (entry point)
├── config.py         # All configurable parameters
├── roostoo_client.py # Authenticated Roostoo REST API client
├── indicators.py     # Technical indicators (SMA, EMA, RSI, MACD, BB, ATR)
├── strategy.py       # Multi-signal strategy → BUY / SELL / HOLD
├── risk_manager.py   # Position tracking, sizing, stop-loss/TP, Sharpe/Sortino/Calmar
├── requirements.txt
├── .env.example
└── tests/
    ├── test_indicators.py
    ├── test_strategy.py
    └── test_risk_manager.py
```

---

## Strategy

The bot uses a **multi-signal hybrid** approach that blends momentum and mean-reversion signals:

| Indicator | Bullish vote (+1) | Bearish vote (−1) |
|---|---|---|
| **RSI (14)** | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **MACD (12/26/9)** | Histogram > 0 | Histogram < 0 |
| **Bollinger Bands (20, 2σ)** | Price in bottom 20 % of band | Price in top 20 % of band |
| **EMA crossover (9/21)** | Fast EMA > Slow EMA | Fast EMA < Slow EMA |

Composite score ∈ [−4, +4]:
- Score ≥ **+2** → **BUY**
- Score ≤ **−2** → **SELL**
- Otherwise → **HOLD**

---

## Risk Management

- **Fixed-fractional sizing** – each trade uses `TRADE_FRACTION` (10 %) of free USD, capped at `MAX_POSITION_FRACTION` (20 %) of total portfolio value.
- **Stop-loss** at −5 % from entry price.
- **Take-profit** at +10 % from entry price.
- **Drawdown circuit-breaker** – trading pauses when portfolio drawdown exceeds 20 %.
- **Performance metrics** logged every 10 iterations: Sharpe ratio, Sortino ratio, Calmar ratio.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A Roostoo API key & secret (apply at [jolly@roostoo.com](mailto:jolly@roostoo.com))

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your API key and secret key
```

### 4. Run the bot

```bash
python bot.py
```

The bot will:
1. Verify connectivity and fetch all tradable pairs from the exchange.
2. Warm up the price history buffer (≥ 40 ticks required before signals fire).
3. Execute trades autonomously every 60 seconds.

### Dry-run mode

To simulate trading without placing real orders:

```python
# In bot.py main() or your own script:
from bot import TradingBot
bot = TradingBot(dry_run=True)
bot.run()
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 68 unit tests cover indicators, strategy logic, and risk management.

---

## Configuration Reference

Edit `config.py` (or override via environment variables) to tune the bot:

| Parameter | Default | Description |
|---|---|---|
| `PRICE_WINDOW` | 60 | Rolling price history length |
| `RSI_PERIOD` | 14 | RSI lookback period |
| `RSI_OVERSOLD` | 30 | RSI buy threshold |
| `RSI_OVERBOUGHT` | 70 | RSI sell threshold |
| `MACD_FAST/SLOW/SIGNAL` | 12/26/9 | MACD parameters |
| `BB_PERIOD` | 20 | Bollinger Bands period |
| `EMA_FAST/SLOW` | 9/21 | EMA crossover periods |
| `MAX_POSITION_FRACTION` | 0.20 | Max portfolio % per asset |
| `TRADE_FRACTION` | 0.10 | Fraction of free USD per trade |
| `STOP_LOSS_PCT` | 0.05 | Stop-loss threshold (5 %) |
| `TAKE_PROFIT_PCT` | 0.10 | Take-profit threshold (10 %) |
| `MAX_DRAWDOWN_LIMIT` | 0.20 | Circuit-breaker drawdown (20 %) |
| `LOOP_INTERVAL_SECONDS` | 60 | Seconds between iterations |

---

## Roostoo API Reference

- REST base URL: `https://mock-api.roostoo.com`
- Full docs: [Roostoo API Documents](https://github.com/roostoo/Roostoo-API-Documents)
