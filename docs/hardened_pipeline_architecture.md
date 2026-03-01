# System Architecture & Technical Resiliency

This document outlines the mechanisms used to ensure the trading bot operates reliably in live market conditions.

## 1. Connectivity & Reliability

The system is designed to handle network failures autonomously without risking accidental trade execution.

| Mechanism | Implementation | Impact |
| :--- | :--- | :--- |
| **Reconnect Logic** | Exponential backoff (1s - 300s) | Prevents connection "thundering herds" during broker outages. |
| **Watchdog Timer** | 60s Heartbeat Monitor | Triggers "Safety Mode" if data stream lags, automatically flattening positions. |
| **Data Validation** | Pydantic Schemas | Rejects malformed financial data (e.g., negative revenue) at ingestion time. |

## 2. Quantitative Strategy & Selection

The system utilizes advanced econometric models and institutional-grade selection filters.

### Triple-Metric Institutional Selection
The trading universe is refined every 63 trading days through a three-priority filter:
1.  **Mean Return Difference (MRD)**: Prioritizes stocks with the highest historical spread between Bull and Bear state returns (exploitable signal).
2.  **Portfolio Beta Steering**: A greedy algorithm selects stocks to maintain a target portfolio Beta of **0.3** relative to SPY.
3.  **Sharpe Ratio**: Final tie-breaker ensuring the best risk-adjusted historical performance.

### Optimal KAMA-MSR Regime Detector (Pomorski)
Based on Piotr Pomorski's UCL PhD Thesis (2024), the system identifies market regimes through:
1.  **Markov-Switching Regression (MSR)**: Fits a 2-state model to identify Low-Variance vs. High-Variance regimes.
2.  **KAMA Filter Overlay**: An adaptive filter identifies Bullish vs. Bearish direction based on standard deviation offsets.
3.  **Contrarian Mapping**: 
    *   **HV_Bear (High-Var Bearish)**: Mapped to `Bull` (Oversold/Mean-Reversion entry).
    *   **LV_Bull (Low-Var Bullish)**: Mapped to `Bear` (Overbought/Bubble exit).

## 3. Risk Management & Position Sizing

Mathematical safety is prioritized over geometric growth to protect against severe drawdowns.

| Component | Logic | Purpose |
| :--- | :--- | :--- |
| **Position Sizing** | Rolling Fractional Kelly | Estimates edge from realized trades; applies strict equity caps. |
| **Volatility Scaling** | VIX-based Reduction | Decreases risk during periods of high market uncertainty. |
| **Circuit Breaker** | -3.0% Intraday Equity Drop | Automatically closes all positions and cancels pending orders from session start. |
| **Spread Filter** | Spread > 0.05% of Price | Defers order entry until liquidity improves. |
| **Position Limits** | Max 1 pos / symbol | Prevents repeated duplicate buys in the same asset. |
| **Buy Cooldown** | 1-hour per symbol | Prevents high-frequency over-trading in a single symbol. |
