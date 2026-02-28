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
| **Position Sizing** | 0.25x Fractional Kelly | Mitigates the impact of inaccurate "edge" estimates. |
| **Volatility Scaling** | VIX-based Reduction | Decreases risk during periods of high market uncertainty. |
| **Regime Filter** | HMM Detector | Ensures the bot only initiates longs during established bull regimes. |
| **Fast-Exit Overlay** | 5-min SPY Trend Slope | Acts as a high-frequency sensor to exit trades before daily trends confirm a reversal. |

## 3. Performance & Safety Breaks

The system handles high data volume through process isolation and automated circuit breakers.

| Guardrail | Trigger | Action |
| :--- | :--- | :--- |
| **Process Isolation** | Multi-processing (3 nodes) | Eliminates Python GIL contention, decoupling ingestion from computation. |
| **Circuit Breaker** | -3.0% Intraday Equity Drop | Automatically closes all positions and cancels pending orders. |
| **Spread Filter** | Spread > 0.05% of Price | Defers order entry until liquidity improves. |
| **Virtual Stop Monitor** | Real-time Trade Breach | Submits IOC (Immediate or Cancel) market orders to ensure fills during gaps. |
