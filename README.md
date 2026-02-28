# Alpaca Markets Automated Trading Pipeline

An anti-fragile, high-frequency-ready automated trading system built for Alpaca Markets. This pipeline implements a 20-step quantitative execution strategy with a heavy focus on resiliency, mathematical robustness, and risk management.

## Key Features

- **Resilient Connectivity:** WebSocket management with exponential backoff and a 500ms heartbeat watchdog.
- **Multi-Process Architecture:** Decoupled ingestion, calculation, and execution processes to eliminate GIL contention.
- **Advanced Risk Management:** Portfolio-level circuit breakers, virtual stop-loss monitoring, and spread/liquidity gatekeeping.
- **Quantitative & Technical Robustness:** TA-Lib for high-performance indicators, Riskfolio-Lib for advanced optimization, and VectorBT for vectorized backtesting.
- **Fundamental & Sentiment Analysis:** Integrated Edgartools for EDGAR data, FinBERT for sentiment scoring, and Z-score based statistical alerts.

## Architecture

The system is split into three core processes communicating via `multiprocessing.Queue`:
1.  **Ingestion:** I/O-bound asyncio loop for live market data.
2.  **Indicators:** CPU-bound process for signal scoring and NLP sentiment.
3.  **Execution:** I/O-bound process for order routing and risk monitoring.

## Getting Started

### Prerequisites

- Python 3.11+
- Alpaca Markets API Keys (Paper or Live)

### Environment Setup

The project is designed to operate from the shared finance virtual environment:
```bash
source /home/jose/venvs/finance/bin/activate
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jreyes913/trading-bot.git
   cd trading-bot
   ```

2. Install/Update project dependencies in the finance venv:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Configuration

Operational thresholds (VIX baselines, Kelly fractions, circuit breaker limits) are managed in `config/config.yaml`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
