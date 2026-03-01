# System Architecture & Technical Resiliency

This document outlines the mechanisms used to ensure the trading bot operates reliably in live market conditions.

## 1. Connectivity & Reliability

The system is designed to handle network failures autonomously without risking accidental trade execution.

| Mechanism | Implementation | Impact |
| :--- | :--- | :--- |
| **Reconnect Logic** | Exponential backoff (1s - 300s) | Prevents connection "thundering herds" during broker outages. |
| **Watchdog Timer** | 500ms Heartbeat Monitor | Triggers "Safety Mode" if data stream lags, automatically flattening positions. |
| **Data Validation** | Pydantic Schemas | Rejects malformed financial data (e.g., negative revenue) at ingestion time. |

## 2. Risk Management & Position Sizing

Mathematical safety is prioritized over geometric growth to protect against severe drawdowns.

| Component | Logic | Purpose |
| :--- | :--- | :--- |
| **Position Sizing** | Rolling Fractional Kelly | Estimates edge from realized trades; applies strict equity caps. |
| **Volatility Scaling** | VIX-based Reduction | Decreases risk during periods of high market uncertainty. |
| Regime Filter | KAMA-MSR Detector | Adaptive filter + Markov-Switching Regression to identify latent states. |

### KAMA-MSR Methodology (Pomorski)

The system identifies market regimes using a two-stage econometric model:

1.  **Adaptive Filtering (KAMA)**: The Kaufman Adaptive Moving Average filters price noise by adjusting its smoothing constant based on the Efficiency Ratio (ER). This ensures the model reacts to trends while ignoring sideways volatility.
2.  **State Identification (MSR)**: A **Markov-Switching Regression** is fitted to the log-returns of the KAMA-filtered series.
    *   The model assumes the existence of latent regimes (Bull/Bear) where the mean and variance of returns differ.
    *   **Classification**: The system calculates the "Smoothed Marginal Probability" of being in the Bull regime.
    *   **Bull**: $P(\text{Bull}) > 0.7$
    *   **Bear**: $P(\text{Bull}) < 0.3$
    *   **Neutral**: $0.3 \le P(\text{Bull}) \le 0.7$ (Ambiguous state)

| **Fast-Exit Overlay** | 5-min SPY Trend Slope | Acts as a high-frequency sensor to exit trades before daily trends confirm a reversal. |

## 3. Performance & Safety Breaks

The system handles high data volume through process isolation and automated circuit breakers.

| Guardrail | Trigger | Action |
| :--- | :--- | :--- |
| **Process Isolation** | Multi-processing (3 nodes) | Eliminates Python GIL contention, decoupling ingestion from computation. |
| **Circuit Breaker** | -3.0% Intraday Equity Drop | Automatically closes all positions and cancels pending orders from session start. |
| **Spread Filter** | Spread > 0.05% of Price | Defers order entry until liquidity improves. |
| **Position Limits** | Max 1 pos / symbol | Prevents repeated duplicate buys in the same asset. |
| **Buy Cooldown** | 1-hour per symbol | Prevents high-frequency over-trading in a single symbol. |
| **Cache Freshness** | Stale > 24 hours | Validates fundamental data integrity and staleness at startup. |
