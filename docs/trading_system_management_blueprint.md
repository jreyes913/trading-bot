# Trading System Management Blueprint

> **Project Infrastructure, Secrets Management, SOPs & Observability**
> Alpaca Markets 20-Step Automated Trading Pipeline

| Field | Value |
|---|---|
| **Document Title** | Trading System Management Blueprint |
| **Classification** | Internal — Confidential |
| **System** | Alpaca Markets HFT Pipeline (alpaca-py, Python 3.11+) |
| **Scope** | Architecture, Config, Dependencies, SOPs, Observability |
| **Version** | 1.0 — Initial Release |
| **Review Cycle** | Quarterly or following any circuit-breaker event |

> **Purpose:** This blueprint defines the management layer for the 20-step automated trading system: how code is organised, how secrets are secured, how dependencies are controlled, the mandatory Standard Operating Procedures engineers must follow during system events, and how production logs are structured for rapid incident diagnosis.

---

## Table of Contents

1. [System Architecture & Directory Layout](#section-01--system-architecture--directory-layout)
2. [Secrets & Configuration Management](#section-02--secrets--configuration-management)
3. [Dependency Management](#section-03--dependency-management)
4. [Operational SOPs](#section-04--operational-sops--standard-operating-procedures)
5. [Logging & Observability](#section-05--logging--observability)

---

## Section 01 — System Architecture & Directory Layout

### 1.1 Project Root Strategy

The repository is organised into a strict separation-of-concerns hierarchy. Each top-level directory has a single, non-overlapping responsibility. No module imports across architectural boundaries without going through a defined interface (queue, config object, or dependency-injected client).

### 1.2 Directory Layout

| Path / File | Module Owner | Process Type | Responsibility |
|---|---|---|---|
| `trading-system/` | — | — | Repository root |
| `  src/` | Engineering | — | All production Python source code |
| `    ingestion.py` | Infra Team | I/O — asyncio | WebSocket stream manager, heartbeat monitor, reconnection backoff, safety-mode pivot |
| `    indicators.py` | Quant Team | CPU — Process | Signal scoring, indicator pipeline (RSI, MACD, ATR, VWAP), divergence detection, FinBERT sentiment |
| `    execution.py` | Infra Team | I/O — Process | Order routing, spread gatekeeper, virtual stop monitor, portfolio circuit breaker |
| `    models/` | Quant Team | — | Sub-package for all model classes |
| `      gatekeeper.py` | Quant Team | Sync | Pydantic FundamentalSnapshot validator; EDGAR data quality enforcement |
| `      regime.py` | Quant Team | CPU | HMM RegimeDetector, FastExitOverlay (5-min SPY trend) |
| `      dcf.py` | Quant Team | CPU | Monte Carlo DCF valuation engine, Kelly / vol-adjusted sizer |
| `      backtest.py` | Quant Team | CPU | Purged & embargoed cross-validation, walk-forward engine |
| `  config/` | DevOps | — | Non-sensitive runtime configuration |
| `    config.yaml` | DevOps | — | All threshold constants: VIX baselines, Kelly fractions, spread limits, embargo sizes |
| `    logging.yaml` | DevOps | — | Python logging configuration: handlers, formatters, log levels |
| `  data/` | — | — | Local artefact storage (git-ignored) |
| `    edgar/` | — | — | Cached EDGAR 10-K / XBRL parquet files |
| `    rejected_tickers/` | — | — | Gatekeeper audit logs (JSON Lines, date-stamped) |
| `    hmm_models/` | — | — | Persisted HMM joblib dumps, versioned by fit date |
| `  tests/` | Engineering | — | pytest suite: unit, integration, backtest regression |
| `    unit/` | Engineering | — | Pure-function tests (validators, sizing, signal scoring) |
| `    integration/` | Engineering | — | Paper-trading smoke tests against Alpaca sandbox |
| `  .env` | DevOps | — | Secret credentials — **NEVER committed to version control** |
| `  .env.example` | DevOps | — | Template showing required keys with blank values |
| `  requirements.txt` | Engineering | — | Pinned production dependency manifest |
| `  requirements-dev.txt` | Engineering | — | Dev/test tooling: pytest, black, mypy, pre-commit |
| `  .gitignore` | DevOps | — | Excludes `.env`, `data/`, `__pycache__`, `*.pkl`, `*.parquet` |
| `  Makefile` | DevOps | — | Targets: `make run`, `make test`, `make audit`, `make halt` |

### 1.3 Module Responsibilities Detail

#### `src/ingestion.py` — Fail-Safe Data Acquisition

- Owns the `ResilientStreamManager` and `HeartbeatMonitor` classes.
- Subscribes to Alpaca WebSocket bars and trades via `alpaca-py` `StockDataStream`.
- Implements exponential backoff reconnection (1s base, 300s ceiling) on any connection error.
- Writes validated bar dicts to the shared `multiprocessing.Queue` (`bar_queue`) consumed by `indicators.py`.
- On 500ms heartbeat breach: sets `safety_mode` flag, calls REST `close_all_positions`, emits `CRITICAL` log event.

#### `src/indicators.py` — Signal & Sentiment Processing

- CPU-bound process: runs in a dedicated OS process, isolated from I/O processes via `multiprocessing`.
- Consumes `bar_queue`; maintains per-symbol rolling deque buffers (`maxlen=500` bars).
- Computes indicator suite on every new bar: RSI(14), MACD(12,26,9), ATR(14), Bollinger %B, VWAP deviation, Ichimoku position.
- Applies `SentimentChangeDetector` (FinBERT + 2-SD z-score threshold) on new NLP input events.
- Publishes scored signal dicts to `signal_queue` consumed by `execution.py`.

#### `src/execution.py` — Order Routing & Circuit Protection

- I/O-bound process: handles all REST API calls to Alpaca trading endpoints.
- Hosts `ExecutionGatekeeper`: validates bid-ask spread (<0.05%) and ADV liquidity (1% of 20-day ADV) before order submission.
- Hosts `VirtualStopMonitor`: subscribes to trade WebSocket stream and fires IOC market orders on price breach.
- Hosts `PortfolioCircuitBreaker`: polls account equity every 10 seconds; triggers 4-phase halt on -3% intraday drawdown.

#### `src/models/` — Quantitative Model Sub-Package

- **`gatekeeper.py`:** `FundamentalSnapshot` Pydantic model with field-level and cross-field validators. Exports `FundamentalGatekeeper` batch processor.
- **`regime.py`:** `GaussianHMM` wrapper (`RegimeDetector`) + `FastExitOverlay` for 5-minute SPY slope divergence detection.
- **`dcf.py`:** Monte Carlo DCF engine (10,000 paths); fractional Kelly sizer (0.25×) with VIX/ATR linear scaling and ADV cap.
- **`backtest.py`:** `purged_embargo_cv` generator; walk-forward harness; Sharpe/drawdown report generator.

---

## Section 02 — Secrets & Configuration Management

### 2.1 Credential Isolation — `.env` File Strategy

All sensitive credentials are stored exclusively in a `.env` file in the repository root. This file is listed in `.gitignore` and must never be committed to version control. The `python-dotenv` library loads these values at process startup. In production, environment variables are injected by the deployment platform (e.g. AWS Secrets Manager, GitHub Actions secrets, systemd `EnvironmentFile`) rather than shipping a `.env` file to the server.

> ⚠️ **Critical Rule:** If a value could cause financial loss or account compromise if leaked — it belongs in `.env`, not in `config.yaml`, not hard-coded, not in a comment.

### 2.2 `.env.example` — Required Keys Template

```bash
# Alpaca Markets
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # Switch to live URL for production

# Twilio SMS Alerts
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=+1XXXXXXXXXX
TWILIO_TO_NUMBER=+1XXXXXXXXXX

# SMTP Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=
SMTP_FROM=alerts@yourdomain.com
SMTP_TO=engineer@yourdomain.com

# Optional: FRED API (if rate-limited on free tier)
FRED_API_KEY=
```

### 2.3 Threshold Externalisation — `config.yaml`

Non-sensitive operational thresholds are stored in `config/config.yaml`. This allows engineers to tune system behaviour between sessions without touching source code, and enables environment-specific overrides (paper vs. live) via YAML anchors or separate files. Changes to `config.yaml` are committed to version control with a descriptive message documenting the reason for each threshold adjustment.

```yaml
# config/config.yaml

connectivity:
  heartbeat_threshold_ms: 500       # Trigger safety mode above this lag
  ws_base_backoff_s: 1              # Initial WebSocket reconnect wait
  ws_max_backoff_s: 300             # Ceiling for exponential backoff

risk:
  circuit_breaker_drawdown: 0.03    # 3% intraday equity loss triggers halt
  circuit_breaker_poll_s: 10        # Equity check interval in seconds
  max_spread_pct: 0.0005            # 0.05% max bid-ask spread for entry
  adv_position_limit: 0.01          # Max position as fraction of 20-day ADV

sizing:
  kelly_fraction: 0.25              # Fractional Kelly multiplier
  vix_baseline: 20.0                # VIX level at which no penalty applies
  vix_max: 40.0                     # VIX level at which size collapses to 0
  atr_baseline_pct: 0.015           # ATR% baseline (no penalty)
  atr_max_pct: 0.040                # ATR% level at which size collapses to 0

regime:
  hmm_states: 3                     # Bull / Neutral / Bear
  fast_exit_slope_threshold: -0.002 # 5-min SPY slope trigger
  fast_exit_lookback_bars: 12       # 12 x 5min = 1hr window

validation:
  cv_gap_size: 10                   # Purge buffer bars around test boundary
  cv_embargo_pct: 0.01              # Fraction of train set embargoed post-test
  dcf_simulations: 10000            # Monte Carlo paths per valuation
  dcf_undervalue_threshold: 0.65    # Min P(undervalued) to pass Phase 2

sentiment:
  zscore_exit_threshold: -2.0       # SD drop to trigger sentiment exit
  min_history_readings: 60          # Minimum scores before z-score is valid
```

> **Version Control Discipline:** Every change to `config.yaml` must be accompanied by a git commit message in the format: `config: [parameter] changed from X to Y — Reason: [justification]`. This creates an auditable change log of all risk parameter decisions.

---

## Section 03 — Dependency Management

### 3.1 Core Production Stack — `requirements.txt`

All production dependencies are pinned to exact versions using `==` to guarantee reproducibility. Never use `>=` in production. Run `pip-compile` (pip-tools) to regenerate the lockfile after any package addition.

```text
# requirements.txt — Production dependencies (pinned)
# Regenerate with: pip-compile requirements.in --output-file requirements.txt

# ── Broker & Market Data ──────────────────────────────
alpaca-py==0.33.1

# ── Data Handling ─────────────────────────────────────
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0              # Parquet I/O for EDGAR cache

# ── Validation ────────────────────────────────────────
pydantic==2.7.4

# ── Technical Indicators ──────────────────────────────
pandas-ta==0.3.14b
scipy==1.13.1

# ── Portfolio Optimisation ────────────────────────────
PyPortfolioOpt==1.5.5
scikit-learn==1.5.0

# ── Regime & Statistical Models ───────────────────────
hmmlearn==0.3.2

# ── NLP / Sentiment ───────────────────────────────────
transformers==4.41.2
torch==2.3.1                 # CPU-only build recommended for cost

# ── Alerts & Notifications ────────────────────────────
twilio==9.2.3

# ── Configuration & Environment ───────────────────────
python-dotenv==1.0.1
PyYAML==6.0.1

# ── Concurrency ───────────────────────────────────────
# multiprocessing: stdlib — no install required
# asyncio: stdlib — no install required

# ── Data Sources ──────────────────────────────────────
pandas-datareader==0.10.0    # FRED yield curve / VIX
yfinance==0.2.40             # Sector ETF momentum
sec-edgar-downloader==5.0.2  # 10-K / 10-Q filing retrieval

# ── Web Scraping ──────────────────────────────────────
requests==2.32.3
lxml==5.2.2
```

### 3.2 Development & Testing Stack — `requirements-dev.txt`

```text
# requirements-dev.txt — Dev/test tooling (not deployed to production)
-r requirements.txt

pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0
black==24.4.2
mypy==1.10.0
pre-commit==3.7.1
pip-tools==7.4.1
ipykernel==6.29.4            # Jupyter for quant research notebooks
```

### 3.3 Dependency Policy

| Setting | Rule |
|---|---|
| **Python Version** | 3.11 or 3.12 (3.12 recommended — faster asyncio event loop) |
| **Environment** | Dedicated virtualenv per deployment; never use system Python |
| **Install Command** | `pip install --no-deps -r requirements.txt` (enforce pinned versions) |
| **Update Policy** | Dependency updates require a full backtest regression pass before merge |
| **Security Scan** | Run `pip-audit` weekly; critical CVEs require same-day patch and retest |

---

## Section 04 — Operational SOPs — Standard Operating Procedures

> ⚠️ **SOP Compliance:** These procedures are mandatory. Deviation must be documented in the incident log within 1 hour of the event. No system may be restarted without completing the applicable SOP steps and obtaining sign-off from the designated Owner.

---

### SOP-01: Safety Mode Protocol (500ms Heartbeat Breach)

Triggered automatically when the `HeartbeatMonitor` detects no WebSocket bar received for more than 500ms. The system enters safety mode autonomously; this SOP governs the human response and recovery.

| Step | Owner | Action |
|---|---|---|
| 1 | System (Auto) | `HeartbeatMonitor` fires. `safety_mode` flag set to `True`. `CRITICAL` log emitted with timestamp and last-known bar symbol. |
| 2 | System (Auto) | `ingestion.py` calls trading REST API: `cancel_orders()` then `close_all_positions(cancel_orders=True)`. Confirms via response status. |
| 3 | On-Call Engineer | Acknowledge `CRITICAL` log alert within 5 minutes. Open the Alpaca dashboard to verify all positions show as closed and no open orders remain. |
| 4 | On-Call Engineer | Diagnose connectivity root cause: check broker status page (status.alpaca.markets), local network, VPS/cloud instance health dashboard. |
| 5 | On-Call Engineer | If root cause is transient (network blip, broker maintenance): monitor the system. The `ResilientStreamManager` will auto-reconnect with exponential backoff. On reconnect, `safety_mode` clears automatically on first received bar. |
| 6 | On-Call Engineer | If root cause is persistent (>5 min outage) or unknown: do not restart. Log the event in the incident register with timestamp, duration, and positions closed. |
| 7 | On-Call Engineer | After full reconnection and at least 60 seconds of clean bar reception, review the `signal_queue` for any stale signals. Flush queue if age > 30 seconds before re-enabling order submission. |
| 8 | Lead Engineer | Next business day: conduct 15-minute post-mortem. Update heartbeat threshold or backoff parameters in `config.yaml` if warranted. Commit changes with documented rationale. |

---

### SOP-02: Circuit Breaker Recovery (−3% Intraday Drawdown)

Triggered when the `PortfolioCircuitBreaker` detects the session equity has declined 3% or more from its opening value. The 4-phase halt sequence runs automatically. **The system cannot be restarted the same trading day without explicit written approval from a senior engineer.**

| Step | Owner | Action |
|---|---|---|
| 1 | System (Auto) | `PortfolioCircuitBreaker` cancels all orders, closes all positions, sets `halted=True`, sends SMS and email alerts. |
| 2 | On-Call Engineer | Confirm receipt of SMS/email alert. Log response time in incident register. |
| 3 | On-Call Engineer | Open Alpaca dashboard. Manually verify: zero open orders, zero open positions. If any remain, close manually via the dashboard before proceeding. |
| 4 | On-Call Engineer | **Do NOT restart** the trading process the same session. The system must remain halted for the remainder of the calendar trading day. |
| 5 | Lead Engineer | Conduct root cause analysis: review structured logs for the 30 minutes preceding the halt. Identify which positions caused the drawdown and why stop-loss / bracket orders did not limit losses as expected. |
| 6 | Lead Engineer | Review: Was the circuit breaker threshold appropriate? Were signals from a correlated cluster of positions? Was market liquidity abnormal (gap open, news event)? |
| 7 | Lead Engineer | Before next session restart: validate that `config.yaml` thresholds are still appropriate. If a parameter change is needed, deploy to paper trading for a minimum of 2 days before live. |
| 8 | Lead Engineer | Clear the `halted` flag manually by restarting the execution process with the updated configuration. Capture opening equity at the new session start. |
| 9 | Team | File a formal incident report documenting cause, impact (P&L), response timeline, and preventive measures within 24 hours. |

---

### SOP-03: Weekly Data Audit — FundamentalGatekeeper Review

The `FundamentalGatekeeper` writes a JSON Lines audit file to `data/rejected_tickers/` every time it processes EDGAR fundamentals. This SOP defines the weekly review cadence to ensure data quality issues are caught before they silently distort the DCF model or portfolio construction.

| Step | Owner | Action |
|---|---|---|
| 1 | Quant Analyst | Every Monday at 09:00 ET (before market open): open the most recent `rejected_tickers/YYYY-MM-DD.jsonl` file. |
| 2 | Quant Analyst | Count total rejections. If rejection rate exceeds 10% of the universe, escalate to Lead Quant — this indicates a systemic EDGAR data quality problem, not isolated ticker issues. |
| 3 | Quant Analyst | For each rejected ticker, review the `reason` field. Categorise each into: NaN/missing field, negative EV, Inf value, schema mismatch. |
| 4 | Quant Analyst | For tickers with persistent rejections (3+ weeks): check the EDGAR filing directly for that ticker. Determine if the company has a structural data reporting anomaly or if the scraping logic needs updating. |
| 5 | Quant Analyst | Update the `known_exclusions` list in `config.yaml` for tickers that are legitimately un-modelable (e.g. holding companies with no consolidated revenue). This prevents repeated rejection noise. |
| 6 | Quant Analyst | If a new rejection category is observed not handled by the current validator, file a backlog ticket to update the Pydantic model with an additional validator. |
| 7 | Lead Engineer | Review the audit summary report (auto-generated by `make audit`) in the weekly engineering sync. Archive the report to the shared ops folder. |

---

## Section 05 — Logging & Observability

### 5.1 Structured JSON Logging Requirements

All log events at `WARNING` level and above must be emitted as newline-delimited JSON (JSON Lines format). Human-readable plain-text logging is permitted for `DEBUG` and `INFO` levels during development only. In production, all levels emit JSON. This enables log aggregators (Datadog, CloudWatch Logs Insights, Grafana Loki) to parse, filter, and alert on structured fields without regex.

### 5.2 Mandatory JSON Fields — All Events

| JSON Field | Type | Example / Description |
|---|---|---|
| `timestamp` | ISO-8601 string | `2025-10-14T13:42:11.304Z` |
| `level` | string | `CRITICAL` \| `ERROR` \| `WARNING` \| `INFO` \| `DEBUG` |
| `process` | string | `ingestion` \| `indicators` \| `execution` \| `gatekeeper` |
| `event` | string | Machine-readable event code — see Event Taxonomy below |
| `message` | string | Human-readable description of the event |
| `session_id` | UUID string | Unique ID per trading session; set at process start |
| `hostname` | string | Server/instance hostname for multi-node identification |

### 5.3 Extended Fields — CRITICAL Events Only

| JSON Field | Type | Example / Description |
|---|---|---|
| `equity_opening` | float | Account equity at session start (circuit breaker events) |
| `equity_current` | float | Account equity at event time |
| `drawdown_pct` | float | Intraday drawdown as a decimal (e.g. `0.0312` = 3.12%) |
| `ticker` | string | Affected symbol (gatekeeper rejections, order events) |
| `rejection_reason` | string | Pydantic validation error message (gatekeeper events) |
| `spread_pct` | float | Observed bid-ask spread at deferral time |
| `stop_price` | float | Virtual stop level that was breached |
| `trade_price` | float | Live trade price that triggered the virtual stop |
| `ws_lag_ms` | float | Measured WebSocket lag in milliseconds (heartbeat events) |
| `backoff_s` | int | Current reconnection backoff interval in seconds |
| `orders_cancelled` | int | Count of orders cancelled in halt sequence |
| `positions_closed` | int | Count of positions closed in halt sequence |

### 5.4 Critical Event Taxonomy

Use the exact event codes below when constructing log entries. Alerting rules in the log aggregator are keyed to these codes — non-standard codes will not trigger alerts.

| Event Code | Description |
|---|---|
| `WS_HEARTBEAT_BREACH` | WebSocket lag exceeded 500ms threshold; safety mode activated |
| `WS_RECONNECT_ATTEMPT` | Attempting WebSocket reconnection; includes attempt number and `backoff_s` |
| `WS_RECONNECT_SUCCESS` | WebSocket reconnected cleanly; backoff reset to base |
| `SAFETY_MODE_ACTIVATED` | System entered safety mode; positions flatten initiated |
| `SAFETY_MODE_CLEARED` | First bar received after safety mode; normal operation resumed |
| `CIRCUIT_BREAKER_HALT` | Portfolio drawdown threshold breached; 4-phase halt initiated |
| `ORDERS_CANCELLED` | All open orders cancelled as part of halt or safety sequence |
| `POSITIONS_CLOSED` | All positions closed via REST API |
| `ALERT_EMAIL_SENT` | Emergency email dispatched; includes recipient address |
| `ALERT_SMS_SENT` | Emergency SMS dispatched via Twilio |
| `ORDER_DEFERRED` | Order deferred due to spread exceeding `MAX_SPREAD_PCT` limit |
| `VIRTUAL_STOP_TRIGGERED` | Live trade breached stop price; IOC market order submitted |
| `TICKER_REJECTED` | `FundamentalGatekeeper` rejected ticker; includes `rejection_reason` |
| `SENTIMENT_EXIT_ALERT` | FinBERT z-score below −2.0 SD; exit signal generated for ticker |
| `FAST_EXIT_TRIGGERED` | HMM says Bull but SPY 5-min slope breached threshold; overlay fired |
| `REGIME_CHANGED` | HMM predicted regime changed state; includes `new_state` label |

### 5.5 Logging Configuration — `logging.yaml`

```yaml
# config/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    (): pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(timestamp)s %(level)s %(process)s %(event)s %(message)s'
  plain:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: plain
    stream: ext://sys.stdout
  file_json:
    class: logging.handlers.TimedRotatingFileHandler
    level: WARNING
    formatter: json
    filename: logs/trading.jsonl
    when: midnight
    backupCount: 90          # 90 days of log retention
    encoding: utf-8
  critical_file:
    class: logging.handlers.TimedRotatingFileHandler
    level: CRITICAL
    formatter: json
    filename: logs/critical.jsonl
    when: midnight
    backupCount: 365         # 1 year of critical event retention
    encoding: utf-8

root:
  level: DEBUG
  handlers: [console, file_json, critical_file]
```

### 5.6 Observability Alerting Policy

| Event Code | Severity | Alert Action |
|---|---|---|
| `CIRCUIT_BREAKER_HALT` | P0 — Critical | PagerDuty page to on-call engineer; SMS + Email (redundant to in-system alert) |
| `SAFETY_MODE_ACTIVATED` | P1 — High | Slack `#trading-alerts` webhook; email to team distribution list |
| `WS_RECONNECT_ATTEMPT` | P2 — Medium | Slack notification if attempt >= 3 within 10 minutes |
| `TICKER_REJECTED` | P3 — Low | Aggregated into daily digest email; no immediate page |
| `ORDER_DEFERRED` | P3 — Low | Aggregated count logged; alert if >20 deferrals in 1 hour |
| `VIRTUAL_STOP_TRIGGERED` | P1 — High | Immediate Slack message with symbol, `stop_price`, `trade_price` |
| `SENTIMENT_EXIT_ALERT` | P2 — Medium | Slack notification with ticker, `z_score`, `baseline_mean` |

### 5.7 Makefile Operational Targets

```makefile
# Makefile

run:          ## Start all three pipeline processes
	python -m src.main

test:         ## Run full pytest suite with coverage report
	pytest tests/ --cov=src --cov-report=term-missing -v

audit:        ## Parse rejected_ticker logs and print weekly summary
	python scripts/audit_gatekeeper.py --week

halt:         ## Emergency manual halt (calls REST flatten, logs MANUAL_HALT)
	python scripts/emergency_halt.py

config-check: ## Validate config.yaml schema before deployment
	python scripts/validate_config.py

backtest:     ## Run walk-forward backtest with purged CV (last 2 years)
	python -m src.models.backtest --years 2

lint:         ## Black + mypy type check
	black src/ tests/ && mypy src/

deps-audit:   ## Scan dependencies for known CVEs
	pip-audit -r requirements.txt
```

---

> **Document Control:** This blueprint is a living document. All engineers are expected to propose amendments via pull request when operational experience reveals gaps. The Lead Engineer is responsible for quarterly review and version increment. Changes to SOPs require acknowledgement from all named process owners before the PR is merged.
