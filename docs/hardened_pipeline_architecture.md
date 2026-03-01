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
| Regime Filter | KAMA-MSR Detector | Uses an adaptive filter to distinguish between trending and sideways regimes. |

### KAMA-MSR Methodology (Pomorski)

The system identifies market regimes using the Kaufman Adaptive Moving Average (KAMA) and the Mean Square Ratio (MSR):

1.  **Efficiency Ratio (ER)**: Measures the "trendiness" of the market.
    $$ER = \frac{|\text{Total Change over } n \text{ bars}|}{\sum |\text{Individual bar changes}|}$$
2.  **Adaptive Smoothing (SC)**: KAMA automatically speeds up during trends and slows down in noise.
3.  **Mean Square Ratio (MSR)**: Compares the variance of the KAMA signal to the raw price noise.
    $$MSR = \frac{\text{MeanSquare}(\Delta KAMA)}{\text{MeanSquare}(\Delta Price)}$$

**Regime Classification**:
*   **Bull**: $MSR > \text{Threshold}$ AND $KAMA_t > KAMA_{t-1}$
*   **Bear**: $MSR > \text{Threshold}$ AND $KAMA_t < KAMA_{t-1}$
*   **Neutral**: $MSR \le \text{Threshold}$ (Market is dominated by noise)

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
