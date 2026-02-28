# Trading System Management Blueprint

This blueprint outlines the operational management of the automated trading system, focusing on organization, secret management, and standard operating procedures (SOPs).

## 1. Directory Structure & Responsibilities

The system is organized to ensure a strict separation of concerns.

| Location | Component | Role |
| :--- | :--- | :--- |
| `src/` | Core Engine | Contains all strategy, ingestion, and execution logic. |
| `config/` | Operational Settings | Stores thresholds, risk limits, and logging levels. |
| `data/` | Artifact Cache | Persists local data (EDGAR, HMM models, audit logs). |
| `logs/` | System Logging | Stores structured (JSONL) and plain-text system events. |
| `tests/` | QA Suite | Contains unit and integration tests (pytest). |

## 2. Secrets & Configuration Management

| Asset | Location | Security Protocol |
| :--- | :--- | :--- |
| **Credentials** | `.env` | Stored locally; ignored by Git; never committed. |
| **API Keys Template** | `.env.example` | Distributed in the repository as a setup template. |
| **Risk Parameters** | `config/config.yaml` | Committed to Git; used to tune intraday system behavior. |

## 3. Standard Operating Procedures (SOPs)

Standard responses to system-triggered events.

| Event | Protocol |
| :--- | :--- |
| **Connectivity Failure** | `ingestion.py` enters "Safety Mode," closes all positions, and waits for a stable heartbeat signal before resuming. |
| **Drawdown Halt** | System enters a locked state; cancels all pending orders. Manual review is required before the next session. |
| **Data Rejection** | Ticker is logged in `data/rejected_tickers/`. Audit required to determine if it's a structural or transient data issue. |

## 4. Free SMS Alerts via SMTP Gateway

Instead of expensive third-party services, this system uses your email to send text alerts directly to your phone. To configure this, set `SMTP_TO_SMS` in your `.env` to your carrier's specific email gateway:

- **Verizon:** `[number]@vtext.com`
- **AT&T:** `[number]@txt.att.net`
- **T-Mobile:** `[number]@tmomail.net`

## 5. Operation Targets (Makefile)

Common commands for system management.

| Command | Action |
| :--- | :--- |
| `make run` | Starts the multi-process trading engine. |
| `make test` | Executes the full pytest suite with coverage. |
| `make lint` | Runs Black and Mypy for code quality and typing. |
| `make halt` | Triggers a manual emergency position flatten. |
