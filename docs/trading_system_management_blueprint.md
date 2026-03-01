# Trading System Management Blueprint

This blueprint outlines the operational management of the automated trading system, focusing on organization, secret management, and standard operating procedures (SOPs).

## 1. Directory Structure & Responsibilities

The system is organized to ensure a strict separation of concerns.

| Location | Component | Role |
| :--- | :--- | :--- |
| `src/` | Core Engine | Contains all strategy, ingestion, and execution logic. |
| `config/` | Operational Settings | Stores thresholds, risk limits, and selection parameters. |
| `scripts/` | Automation | Logic for fundamental caching and universe refreshes. |
| `data/` | Artifact Cache | Persists local data (MRD metrics, fundamental cache, update logs). |
| `logs/` | System Logging | Stores structured (JSONL) and plain-text system events. |
| `tests/` | QA Suite | Contains unit and integration tests (pytest). |

## 2. Secrets & Configuration Management

| Asset | Location | Security Protocol |
| :--- | :--- | :--- |
| **Credentials** | `.env` | Stored locally; ignored by Git; never committed. |
| **API Keys Template** | `.env.example` | Distributed in the repository as a setup template. |
| **Risk Parameters** | `config/config.yaml` | Committed to Git; used to tune intraday system behavior. |

## 3. Automation Cycles (Cron)

The system maintains its edge through scheduled quantitative updates.

| Task | Command | Schedule | Purpose |
| :--- | :--- | :--- | :--- |
| **Fundamental Cache** | `make cache-fund` | Daily (8:00 AM) | Refreshes DCF inputs (EBITDA, Debt). |
| **Universe Refresh** | `make refresh-universe` | Sat (Quarterly) | Re-optimizes the 40-stock universe for MRD/Beta. |

## 4. Standard Operating Procedures (SOPs)

Standard responses to system-triggered events.

| Event | Protocol |
| :--- | :--- |
| **Connectivity Failure** | `ingestion.py` enters "Safety Mode," closes all positions, and waits for a stable heartbeat signal before resuming. |
| **Drawdown Halt** | System enters a locked state; cancels all pending orders. Manual review is required before the next session. |
| **Data Rejection** | Ticker is logged in `data/rejected_tickers/`. Audit required to determine if it's a structural or transient data issue. |

## 5. Free SMS Alerts via SMTP Gateway

Instead of expensive third-party services, this system uses your email to send text alerts directly to your phone. To configure this, set `SMTP_TO_SMS` in your `.env` to your carrier's specific email gateway (e.g., `[number]@tmomail.net` for T-Mobile).

## 6. Operation Targets (Makefile)

Common commands for system management.

| Command | Action |
| :--- | :--- |
| `make run` | Starts the multi-process trading engine. |
| `make cache-fund` | Manually triggers a fundamental data refresh. |
| `make refresh-universe` | Manually triggers a 40-stock reselection (Takes 10-15 mins). |
| `make test` | Executes the full pytest suite with coverage. |
| `make halt` | Triggers a manual emergency position flatten. |
