# Alpaca Markets Automated Trading Bot

## About This Project
This system represents the implementation of extensive research into retail market analysis across three core disciplines: **Fundamental**, **Quantitative**, and **Technical** analysis. 

While the architectural design and strategic parameters are rooted in my engineering background and market studies, the majority of the codebase was implemented by **Gemini CLI**. As an engineer by trade rather than a professional programmer, I leveraged AI to bridge the gap between quantitative theory and production-grade execution logic.

This is a resilient, automated trading system designed for the Alpaca Markets platform. It implements a 20-step quantitative execution pipeline focused on system reliability, mathematical safety, and comprehensive risk management.

## Core System Features

| Feature | Description |
| :--- | :--- |
| **KAMA-MSR Regime** | Implements the optimal KAMA+MSR framework from Piotr Pomorski's 2024 thesis for high-precision market state detection. |
| **Triple-Metric Selector** | Autonomously selects 40 stocks based on Mean Return Difference (MRD), target Beta (0.3), and Sharpe Ratio. |
| **Resilient Connectivity** | Implements exponential backoff and a 60s heartbeat watchdog to handle network instability and low IEX volume. |
| **Multi-Process Design** | Separates ingestion, signal calculation, and order execution into three distinct OS processes to ensure high performance. |
| **Portfolio Guardrails** | Features a 3% intraday drawdown circuit breaker that automatically flattens positions and halts trading. |
| **Position Sizing** | Uses Rolling Fractional Kelly math adjusted for VIX and ATR volatility to prevent over-leveraging. |
| **Sentiment Analysis** | Utilizes FinBERT (NLP) to analyze market news and prevent entries when sentiment drops below statistical norms. |

## System Architecture Overview

| Component | Responsibility |
| :--- | :--- |
| **Universe Selector** | Filters S&P 500 for "Moat" stocks and optimizes for MRD/Beta every 63 trading days. |
| **Data Ingestion** | Maintains the live WebSocket connection and feeds raw market data into the system. |
| **Signal Calculation** | Processes KAMA-MSR econometric models and technical indicators (RSI, MACD). |
| **Order Execution** | Manages Monte Carlo DCF valuations, Kelly sizing, and spread gatekeeping. |

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
   # Populate .env with your Alpaca, FMP, and SMTP credentials.
   ```

### Free SMS Alerts via SMTP
You can receive SMS alerts for free by sending an email to your carrier's SMS gateway:
- **Verizon:** `[number]@vtext.com`
- **AT&T:** `[number]@txt.att.net`
- **T-Mobile:** `[number]@tmomail.net`

## Automation Cycles

| Task | Frequency | Description |
| :--- | :--- | :--- |
| **Fundamental Cache** | Daily (8:00 AM) | Updates financial data (EBITDA, Debt, etc.) for the active universe. |
| **Universe Refresh** | Every 63 Trading Days | Re-evaluates the S&P 500 for Moat and MRD/Beta optimization. |

## Configuration
All operational limits, such as risk thresholds and bet sizes, are externalized in `config/config.yaml` for easy adjustment without modifying code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
