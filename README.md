# Alpaca Markets Automated Trading Bot

This is a resilient, automated trading system designed for the Alpaca Markets platform. It implements a 20-step quantitative execution pipeline focused on system reliability, mathematical safety, and comprehensive risk management.

## Core System Features

| Feature | Description |
| :--- | :--- |
| **Resilient Connectivity** | Implements exponential backoff and a 500ms heartbeat watchdog to handle network instability. |
| **Multi-Process Design** | Separates ingestion, signal calculation, and order execution into three distinct OS processes to ensure high performance. |
| **Portfolio Guardrails** | Features a 3% intraday drawdown circuit breaker that automatically flattens positions and halts trading. |
| **Position Sizing** | Uses Fractional Kelly math adjusted for VIX and ATR volatility to prevent over-leveraging. |
| **Sentiment Analysis** | Utilizes FinBERT (NLP) to analyze market news and prevent entries when sentiment drops below statistical norms. |

## System Architecture Overview

| Component | Responsibility |
| :--- | :--- |
| **Data Ingestion** | Maintains the live WebSocket connection and feeds raw market data into the system. |
| **Signal Calculation** | Processes technical indicators (RSI, MACD) and runs sentiment analysis models. |
| **Order Execution** | Manages order routing, spread gatekeeping, and virtual stop-loss monitoring. |

## Getting Started

### Prerequisites
- Python 3.11+
- Alpaca Markets API Keys (Paper or Live)
- A local environment with the required financial libraries installed.

### Environment Setup
This project operates within a dedicated finance virtual environment:
```bash
source /home/jose/venvs/finance/bin/activate
```

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/jreyes913/trading-bot.git
   cd trading-bot
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure secrets:**
   ```bash
   cp .env.example .env
   # Populate .env with your Alpaca and alert credentials.
   ```

## Configuration
All operational limits, such as risk thresholds and bet sizes, are externalized in `config/config.yaml` for easy adjustment without modifying code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
