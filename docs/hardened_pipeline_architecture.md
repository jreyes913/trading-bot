# Hardened Trading Pipeline Architecture

> **Architectural Specifications & Code Patterns for Anti-Fragile Execution**
> Alpaca Markets 20-Step Automated Trading Pipeline

| Field | Value |
|---|---|
| **Prepared for** | Quantitative Systems Engineering Team |
| **System** | Alpaca Markets 20-Step Automated Trading Pipeline |
| **Focus** | Resiliency · Model Robustness · Execution Safety · Circuit Breakers |
| **Classification** | Internal — Confidential |

> ⚠️ All code patterns are production-oriented and must be reviewed before deployment.

---

## Table of Contents

1. [Technical Resiliency & Anti-Fragility](#section-1--technical-resiliency--anti-fragility)
   - [1A. Fail-Safe Connectivity](#1a-fail-safe-connectivity)
   - [1B. Data Validation Gatekeeper](#1b-data-validation-gatekeeper-edgar--pydantic)
   - [1C. Latency Optimization — Multi-Process Architecture](#1c-latency-optimization--multi-process-architecture)
2. [Mathematical & Model Robustness](#section-2--mathematical--model-robustness)
   - [2A. Fractional Kelly + Volatility-Adjusted Sizing](#2a-fractional-kelly--volatility-adjusted-sizing)
   - [2B. Regime Lag Mitigation — Fast-Exit Overlay](#2b-regime-lag-mitigation--fast-exit-overlay)
   - [2C. Backtest De-Biasing — Purged & Embargoed CV](#2c-backtest-de-biasing--purged--embargoed-cross-validation)
3. [Execution & Risk Circuit Breakers](#section-3--execution--risk-circuit-breakers)
   - [3A. Slippage, Liquidity Modeling & Virtual Stop-Loss](#3a-slippage-liquidity-modeling--virtual-stop-loss)
   - [3B. Sentiment Nuance — Change-of-Sentiment Trigger](#3b-sentiment-nuance--change-of-sentiment-trigger)
4. [Global Kill-Switches](#section-4--global-kill-switches)
   - [4A. Portfolio-Level Circuit Breaker](#4a-portfolio-level-circuit-breaker)

---

## Section 1 — Technical Resiliency & Anti-Fragility

### 1A. Fail-Safe Connectivity

#### System Design Pattern

The WebSocket connection is wrapped in a `ResilientStreamManager` that implements three interlocking mechanisms:

- **Exponential Backoff Reconnection:** On any WebSocket error or clean disconnect, the system waits 1s, 2s, 4s, 8s… up to a configurable `MAX_BACKOFF` of 300 seconds. The backoff resets to baseline on a successful reconnect. This prevents thundering-herd reconnect storms during broker outages.

- **HeartbeatMonitor (500ms Threshold):** A lightweight monotonic-clock tracker is updated on every incoming bar event. A separate `asyncio` watchdog task wakes every 100ms and checks if the last pulse is older than 500ms. On stale detection, it atomically sets a `safety_mode` flag.

- **Automatic Safety Pivot:** When `safety_mode` is `True`, the system immediately calls the Alpaca REST API to cancel all open orders and close all positions. The system logs a `CRITICAL` event and waits for the next incoming bar to auto-exit safety mode, preventing false positives from brief network hiccups.

> **Architecture Flow:**
> `WS Bar Event → HeartbeatMonitor.pulse() → Watchdog (100ms poll) → [if stale > 500ms] → Safety Mode → Flatten Positions + Log → Auto-recover on next bar`

#### Python Code Snippet

```python
import asyncio, time, logging
from collections import deque
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)

class HeartbeatMonitor:
    """Tracks last message timestamp; fires alert if > 500ms stale."""
    def __init__(self, threshold_ms: float = 500.0):
        self.threshold_ms = threshold_ms
        self.last_tick = time.monotonic()

    def pulse(self):
        self.last_tick = time.monotonic()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_tick) * 1000 > self.threshold_ms

class ResilientStreamManager:
    MAX_BACKOFF = 300   # seconds
    BASE_BACKOFF = 1

    def __init__(self, api_key, secret_key, symbols, trading_client):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols
        self.trading = trading_client
        self.hb = HeartbeatMonitor(threshold_ms=500)
        self._bar_buffer: deque = deque(maxlen=500)
        self._attempt = 0
        self._safety_mode = False

    async def _on_bar(self, bar):
        self.hb.pulse()
        self._bar_buffer.append(bar)
        if self._safety_mode:
            logger.warning("Bar received; exiting safety mode.")
            self._safety_mode = False

    async def _watchdog(self):
        """Continuously checks heartbeat; triggers safety mode on lag."""
        while True:
            await asyncio.sleep(0.1)          # check every 100ms
            if self.hb.is_stale() and not self._safety_mode:
                logger.critical("STREAM LAG > 500ms — entering Safety Mode")
                self._safety_mode = True
                await self._flatten_all_positions()

    async def _flatten_all_positions(self):
        """Emergency flatten: cancel orders then close all positions."""
        try:
            self.trading.cancel_orders()
            self.trading.close_all_positions(cancel_orders=True)
            logger.critical("All positions flattened via Safety Mode.")
        except Exception as e:
            logger.error(f"Flatten error: {e}")

    async def _connect(self):
        wss = StockDataStream(self.api_key, self.secret_key)
        wss.subscribe_bars(self._on_bar, *self.symbols)
        await wss._run_forever()

    async def run(self):
        asyncio.create_task(self._watchdog())
        backoff = self.BASE_BACKOFF
        while True:
            try:
                self._attempt += 1
                logger.info(f"WS connect attempt #{self._attempt}")
                await self._connect()
            except Exception as e:
                logger.error(f"WS error: {e}. Retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)
            else:
                backoff = self.BASE_BACKOFF   # reset on clean disconnect
```

---

### 1B. Data Validation Gatekeeper (EDGAR / Pydantic)

#### System Design Pattern

All raw EDGAR fundamentals must pass through a Pydantic validation layer before reaching the DCF model. This eliminates silent NaN propagation and impossible financial constructs from corrupting downstream valuations.

- **Field-Level Validators:** Revenue, Shares Outstanding, and Market Cap must be positive finite numbers. EBIT, Debt, and Cash must be finite (NaN/Inf rejected). Pydantic's `@validator` decorator enforces this at instantiation time.

- **Enterprise Value Cross-Check:** A `@root_validator` computes `EV = Market Cap + Total Debt − Cash` and rejects the entire record if `EV < 0`. A negative EV is an accounting impossibility under normal conditions and signals corrupt source data.

- **Audit Trail:** Rejected tickers are logged to a `self.rejected` list with the exact validation error message, enabling a daily audit report of data quality issues per EDGAR filing.

> ⚠️ **Risk:** A single NaN propagating into the covariance matrix (Step 10) can silently corrupt the entire portfolio optimization, producing infinite or zero weights. This validator is non-optional.

#### Python Code Snippet

```python
from pydantic import BaseModel, validator, root_validator
from typing import Optional
import math

class FundamentalSnapshot(BaseModel):
    ticker: str
    revenue: float
    ebit: float
    total_debt: float
    cash: float
    shares_outstanding: float
    market_cap: float
    ebitda: Optional[float] = None

    @validator('revenue', 'shares_outstanding', 'market_cap')
    def must_be_positive(cls, v, field):
        if math.isnan(v) or v <= 0:
            raise ValueError(f"{field.name} must be a positive finite number, got {v}")
        return v

    @validator('ebit', 'total_debt', 'cash', always=True)
    def must_be_finite(cls, v, field):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{field.name} contains NaN or Inf: {v}")
        return v

    @root_validator
    def validate_enterprise_value(cls, values):
        mc = values.get('market_cap', 0)
        debt = values.get('total_debt', 0)
        cash = values.get('cash', 0)
        ev = mc + debt - cash
        if ev < 0:
            raise ValueError(
                f"Negative Enterprise Value detected (EV={ev:.0f}). "
                f"Likely data error — rejecting ticker."
            )
        return values

class FundamentalGatekeeper:
    def __init__(self):
        self.rejected: list[dict] = []
        self.accepted: list[FundamentalSnapshot] = []

    def process(self, raw_records: list[dict]) -> list[FundamentalSnapshot]:
        for rec in raw_records:
            try:
                snap = FundamentalSnapshot(**rec)
                self.accepted.append(snap)
            except Exception as e:
                self.rejected.append({"ticker": rec.get("ticker"), "reason": str(e)})
                continue
        return self.accepted
```

---

### 1C. Latency Optimization — Multi-Process Architecture

#### System Design Pattern

The Python GIL prevents true CPU parallelism within a single `asyncio` loop. The solution is to split the pipeline into three separate OS-level processes communicating via `multiprocessing.Queue`:

- **Process 1 — Data Ingestion (I/O bound):** Runs its own `asyncio` event loop. Receives WebSocket bars and places raw bar dicts onto `bar_queue`. No computation; minimal latency.

- **Process 2 — Indicator Calculation (CPU bound):** Consumes `bar_queue` via a blocking `.get()` call. Maintains per-symbol rolling deques and runs `pandas-ta` indicator computations. Places scored signals onto `signal_queue`. CPU contention is fully isolated.

- **Process 3 — Order Execution (network I/O):** Consumes `signal_queue` and makes REST API calls to Alpaca. Execution latency is decoupled from indicator computation, preventing stale signals from accumulating behind a slow order submission.

#### Python Code Snippet

```python
import multiprocessing as mp
from multiprocessing import Queue, Process
import time

# --- Process 1: Data Ingestion (I/O bound — asyncio inside process) ---
def ingestion_process(bar_queue: Queue, api_key: str, secret_key: str, symbols: list):
    import asyncio
    from alpaca.data.live import StockDataStream

    async def on_bar(bar):
        bar_queue.put_nowait({"symbol": bar.symbol, "close": bar.close,
                              "volume": bar.volume, "timestamp": bar.timestamp})

    async def run():
        wss = StockDataStream(api_key, secret_key)
        wss.subscribe_bars(on_bar, *symbols)
        await wss._run_forever()

    asyncio.run(run())

# --- Process 2: Indicator Calculation (CPU bound) ---
def indicator_process(bar_queue: Queue, signal_queue: Queue):
    from collections import defaultdict, deque
    import pandas as pd, pandas_ta as ta

    buffers = defaultdict(lambda: deque(maxlen=200))
    while True:
        bar = bar_queue.get()          # blocking get — no GIL contention
        sym = bar["symbol"]
        buffers[sym].append(bar["close"])
        if len(buffers[sym]) < 30:
            continue
        s = pd.Series(list(buffers[sym]))
        rsi = ta.rsi(s, length=14).iloc[-1]
        macd_line = ta.macd(s)["MACD_12_26_9"].iloc[-1]
        score = (1 if rsi < 70 else -1) + (1 if macd_line > 0 else -1)
        signal_queue.put({"symbol": sym, "score": score,
                          "price": bar["close"], "ts": bar["timestamp"]})

# --- Process 3: Order Execution (network I/O) ---
def execution_process(signal_queue: Queue, api_key: str, secret_key: str):
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    client = TradingClient(api_key, secret_key, paper=True)
    while True:
        sig = signal_queue.get()
        if sig["score"] >= 2:
            req = MarketOrderRequest(symbol=sig["symbol"], qty=1,
                                     side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            client.submit_order(req)

def launch_pipeline(api_key, secret_key, symbols):
    bar_q, sig_q = Queue(maxsize=1000), Queue(maxsize=500)
    procs = [
        Process(target=ingestion_process, args=(bar_q, api_key, secret_key, symbols)),
        Process(target=indicator_process,  args=(bar_q, sig_q)),
        Process(target=execution_process,  args=(sig_q, api_key, secret_key)),
    ]
    for p in procs:
        p.daemon = True; p.start()
    for p in procs:
        p.join()
```

---

## Section 2 — Mathematical & Model Robustness

### 2A. Fractional Kelly + Volatility-Adjusted Sizing

#### System Design Pattern

Full Kelly sizing is known to produce catastrophic drawdowns when the edge estimate is even slightly optimistic. The hardened approach uses three compounding constraints:

- **Fractional Kelly (0.25×):** The position fraction is capped at 25% of the theoretically optimal Kelly fraction. This reduces both the maximum position size and the variance of outcomes, at the cost of slower geometric growth — an acceptable trade-off for a retail system.

- **Volatility Scaling via VIX:** When VIX is at or below baseline (20), no penalty is applied. As VIX rises linearly toward the maximum threshold (40), the position size scales linearly toward zero. This creates automatic de-risking during regime stress.

- **Volatility Scaling via ATR%:** The same linear scaling applies using the stock's own ATR as a percentage of price. This accounts for idiosyncratic volatility independent of the macro VIX level.

- **ADV Liquidity Cap:** No position may exceed 1% of the 20-day Average Daily Volume. This prevents the system from taking positions it cannot exit without moving the market.

> **Formula:**
> `final_f = min(vix_scale, atr_scale) × fractional_kelly(edge, odds, 0.25)`
> `shares = min(equity × final_f / price, adv_20 × 0.01)`

#### Python Code Snippet

```python
import numpy as np

def fractional_kelly(edge: float, odds: float, fraction: float = 0.25) -> float:
    """Full Kelly = edge/odds. Fractional Kelly scales by 'fraction'."""
    full_kelly = edge / odds if odds > 0 else 0.0
    return max(0.0, full_kelly * fraction)

def volatility_adjusted_size(
    base_fraction: float,
    current_vix: float,
    current_atr_pct: float,          # ATR as % of price
    vix_baseline: float = 20.0,
    vix_max: float = 40.0,
    atr_baseline: float = 0.015,
    atr_max: float = 0.04
) -> float:
    """
    Linearly reduce position size as VIX or ATR rises above baseline.
    At vix_max or atr_max, scale collapses to 0.
    """
    vix_scale = max(0.0, 1.0 - (current_vix - vix_baseline) / (vix_max - vix_baseline))
    atr_scale = max(0.0, 1.0 - (current_atr_pct - atr_baseline) / (atr_max - atr_baseline))
    combined_scale = min(vix_scale, atr_scale)   # most conservative wins
    return base_fraction * combined_scale

def compute_position_size(
    account_equity: float,
    price: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_vix: float,
    atr_pct: float,
    adv_20: float,                   # 20-day Average Daily Volume in shares
    fraction: float = 0.25
) -> int:
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    odds = avg_win / avg_loss if avg_loss > 0 else 1.0
    kelly_f = fractional_kelly(edge, odds, fraction)
    vol_f   = volatility_adjusted_size(kelly_f, current_vix, atr_pct)
    dollar_size = account_equity * vol_f
    shares = int(dollar_size / price)

    # Liquidity constraint: cap at 1% of 20-day ADV
    max_adv_shares = int(adv_20 * 0.01)
    shares = min(shares, max_adv_shares)
    return max(0, shares)
```

---

### 2B. Regime Lag Mitigation — Fast-Exit Overlay

#### System Design Pattern

Hidden Markov Models are trained on daily features and thus lag intraday market turning points by hours. The Fast-Exit Overlay is a real-time override layer:

- **HMM Regime as Background State:** The daily HMM prediction (Bull/Neutral/Bear) is computed once at market open and used to gate new long entries throughout the session.

- **5-Minute SPY Trend as Fast Sensor:** A linear regression slope is computed over the last 12 five-minute SPY bars (approximately one hour of price action). The slope is normalized by mean price to produce a scale-invariant rate of change.

- **Divergence Detection:** If the HMM reports "Bull" but the SPY 5-minute slope drops below a configurable threshold (e.g., −0.002, meaning a sharp intraday sell-off), the overlay fires a fast-exit signal. This bridges the lag between daily regime changes and intraday execution.

#### Python Code Snippet

```python
import numpy as np
import pandas as pd
from hmmlearn import hmm

class RegimeDetector:
    def __init__(self, n_states: int = 3):
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                                      n_iter=200, random_state=42)
        self.bull_state = None   # identified post-fit

    def fit(self, features: np.ndarray):
        self.model.fit(features)
        means = self.model.means_[:, 0]   # assume first feature is return-like
        self.bull_state = int(np.argmax(means))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def current_state(self, latest_features: np.ndarray) -> str:
        state = self.model.predict(latest_features.reshape(1, -1))[0]
        return "Bull" if state == self.bull_state else "Other"

class FastExitOverlay:
    """
    Monitors 5-min SPY bars. If SPY trend diverges sharply from
    HMM 'Bull' state, triggers early exit regardless of daily regime.
    """
    def __init__(self, spy_lookback: int = 12, slope_threshold: float = -0.002):
        self.lookback = spy_lookback         # 12 x 5min = 1hr
        self.slope_threshold = slope_threshold

    def spy_slope(self, spy_closes: pd.Series) -> float:
        closes = spy_closes.iloc[-self.lookback:]
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes.values, 1)
        normalized = slope / closes.mean()
        return normalized

    def should_fast_exit(self, regime: str, spy_closes: pd.Series) -> bool:
        if regime != "Bull":
            return False
        slope = self.spy_slope(spy_closes)
        if slope < self.slope_threshold:
            return True   # HMM says Bull, but intraday SPY is dropping sharply
        return False
```

---

### 2C. Backtest De-Biasing — Purged & Embargoed Cross-Validation

#### System Design Pattern

Standard k-fold cross-validation on time-series data allows future information to leak into training sets via overlapping label windows and correlated features. The Purged + Embargo method eliminates both leak pathways:

- **Purge Buffer:** A configurable `gap_size` (e.g., 10 bars) of training samples immediately adjacent to the test period boundary is removed from training. This prevents training on labels that overlap with the test period's features.

- **Embargo Zone:** A fraction of the training set immediately following the test period is excluded. This prevents lagging features (e.g., a 20-day rolling average) from carrying test-period information back into training.

- **Non-Random Splits:** Folds are constructed as contiguous time blocks, never shuffled. Shuffling destroys the temporal ordering that the embargo and purge are designed to protect.

> ⚠️ **Risk:** Traditional `sklearn` KFold on time-series data produces Sharpe ratios that are inflated by 20–50% in backtests relative to live performance. This is the primary source of "P-hacking" in systematic strategies.

#### Python Code Snippet

```python
import numpy as np

def purged_embargo_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    gap_size: int = 10,         # bars to purge around boundary
    embargo_pct: float = 0.01   # fraction of training set to embargo after test
):
    """
    Yields (train_idx, test_idx) with purge + embargo gaps.

    Purge  : removes training samples adjacent to test period (leakage prevention).
    Embargo: removes training samples immediately AFTER test period
             (prevents look-ahead from lagging features).
    """
    n = len(X)
    fold_size = n // n_splits
    embargo_size = int(n * embargo_pct)

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end   = test_start + fold_size

        # Purge: exclude samples within gap_size of the test boundary
        purge_start = max(0,  test_start - gap_size)
        purge_end   = min(n,  test_end   + gap_size)

        # Embargo: exclude samples immediately after test set
        embargo_end = min(n, test_end + embargo_size)

        # Build train index: all EXCEPT purged + embargoed
        all_idx   = np.arange(n)
        excluded  = set(range(purge_start, embargo_end))
        train_idx = np.array([i for i in all_idx if i not in excluded])
        test_idx  = np.arange(test_start, test_end)

        yield train_idx, test_idx
```

---

## Section 3 — Execution & Risk Circuit Breakers

### 3A. Slippage, Liquidity Modeling & Virtual Stop-Loss

#### System Design Pattern

Two distinct execution risks require separate handling: pre-trade spread validation and post-entry stop-loss resilience during price gaps.

- **Max Spread Check (Pre-Trade Gate):** Before any order is submitted, the live bid-ask spread is queried via the Alpaca data client. If the spread exceeds 0.05% of the mid-price, the order is deferred and logged. Wide spreads indicate low liquidity, wide market-maker uncertainty, or pre-announcement volatility — all conditions where entry is disadvantageous.

- **ADV Liquidity Constraint:** No position may exceed 1% of the 20-day Average Daily Volume. Enforced at the position sizing stage, before the spread check.

- **Virtual Stop Monitor:** Broker-side bracket orders are susceptible to failure during price gaps (e.g., overnight gaps, circuit-breaker halts). The `VirtualStopMonitor` subscribes to the WebSocket trade stream and monitors each trade tick. If the live trade price breaches the stop level, the monitor immediately submits an `IOC` (Immediate-or-Cancel) market order. Unlike a limit order, IOC ensures execution at whatever price is available rather than missing the fill entirely.

#### Python Code Snippet

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class ExecutionGatekeeper:
    MAX_SPREAD_PCT = 0.0005   # 0.05% max allowed spread

    def __init__(self, trading_client: TradingClient, data_client):
        self.trading = trading_client
        self.data = data_client

    def get_quote(self, symbol: str) -> dict:
        quote = self.data.get_stock_latest_quote(symbol)
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0
        return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}

    def spread_ok(self, symbol: str) -> tuple[bool, dict]:
        q = self.get_quote(symbol)
        ok = q["spread_pct"] <= self.MAX_SPREAD_PCT
        return ok, q

    def submit_if_liquid(self, symbol: str, qty: int, side: OrderSide):
        ok, quote = self.spread_ok(symbol)
        if not ok:
            print(f"[DEFERRED] {symbol}: spread {quote['spread_pct']*100:.3f}% > limit. Deferring order.")
            return None
        req = MarketOrderRequest(symbol=symbol, qty=qty,
                                 side=side, time_in_force=TimeInForce.DAY)
        return self.trading.submit_order(req)


class VirtualStopMonitor:
    """
    Monitors WebSocket stream. If price breaches stop_price,
    submits an IOC market order to exit immediately.
    """
    def __init__(self, trading_client, symbol: str, stop_price: float,
                 side: str = "sell", qty: int = 1):
        self.trading = trading_client
        self.symbol = symbol
        self.stop_price = stop_price
        self.side = side
        self.qty = qty
        self.triggered = False

    def on_trade(self, trade):
        if self.triggered:
            return
        price = float(trade.price)
        breach = (self.side == "sell" and price <= self.stop_price) or \
                 (self.side == "buy"  and price >= self.stop_price)
        if breach:
            self.triggered = True
            self._submit_ioc()

    def _submit_ioc(self):
        side = OrderSide.SELL if self.side == "sell" else OrderSide.BUY
        req = MarketOrderRequest(symbol=self.symbol, qty=self.qty,
                                 side=side, time_in_force=TimeInForce.IOC)
        try:
            self.trading.submit_order(req)
            print(f"[VIRTUAL STOP] IOC submitted for {self.symbol} @ stop {self.stop_price}")
        except Exception as e:
            print(f"[ERROR] Virtual stop IOC failed: {e}")
```

---

### 3B. Sentiment Nuance — Change-of-Sentiment Trigger

#### System Design Pattern

Binary positive/negative FinBERT output generates excessive false-positive exit signals because it ignores each company's baseline sentiment distribution. The Change-of-Sentiment Trigger resolves this by applying statistical process control logic:

- **Continuous Sentiment Scoring:** FinBERT returns a label (positive/negative/neutral) and a confidence score. These are mapped to a continuous scale: `+confidence` for positive, `−confidence` for negative, `0` for neutral.

- **Rolling Historical Baseline:** Per-ticker, a rolling deque stores up to 756 sentiment scores (approximately 3 years of daily readings). The mean and standard deviation of this history define the company's "normal" sentiment distribution.

- **Z-Score Threshold:** A new sentiment reading is only flagged as an exit alert if its z-score relative to the historical baseline is below −2.0. This corresponds to a genuine anomalous negative shift, not normal sentiment noise. Single article bias is absorbed by the baseline without triggering unnecessary exits.

#### Python Code Snippet

```python
import numpy as np
from transformers import pipeline
from collections import deque

class SentimentChangeDetector:
    """
    Tracks rolling sentiment score per ticker.
    Only triggers exit if score drops > 2 std devs below 3-yr baseline.
    """
    def __init__(self, history_window: int = 756):   # ~3 years of trading days
        self.finbert = pipeline("sentiment-analysis",
                                 model="ProsusAI/finbert", truncation=True)
        self.history: dict[str, deque] = {}

    def _score(self, text: str) -> float:
        result = self.finbert(text[:512])[0]
        label, conf = result["label"], result["score"]
        return conf if label == "positive" else (-conf if label == "negative" else 0.0)

    def update(self, ticker: str, new_text: str) -> dict:
        score = self._score(new_text)
        if ticker not in self.history:
            self.history[ticker] = deque(maxlen=756)
        self.history[ticker].append(score)
        hist = np.array(self.history[ticker])

        if len(hist) < 60:    # need minimum history
            return {"ticker": ticker, "score": score, "signal": "INSUFFICIENT_HISTORY"}

        mean_hist = np.mean(hist[:-1])   # exclude current
        std_hist  = np.std(hist[:-1])
        z_score   = (score - mean_hist) / std_hist if std_hist > 0 else 0.0

        signal = "EXIT_ALERT" if z_score < -2.0 else "HOLD"
        return {
            "ticker": ticker, "score": round(score, 4),
            "z_score": round(z_score, 4), "signal": signal,
            "baseline_mean": round(mean_hist, 4),
            "baseline_std": round(std_hist, 4)
        }
```

---

## Section 4 — Global Kill-Switches

### 4A. Portfolio-Level Circuit Breaker

#### System Design Pattern

The circuit breaker runs as a continuously polling background coroutine. It is architecturally independent of all signal, indicator, and execution logic. Its four-phase halt sequence is strictly ordered:

1. **Trigger Condition:** Opening equity is captured at system start via the Alpaca REST API. Every poll interval (default: 10 seconds), current account equity is re-fetched. If the intraday drawdown exceeds 3%, the halt sequence fires atomically.

2. **Step 1 — Cancel All Open Orders:** `trading.cancel_orders()` is called first. This prevents any pending orders from filling after the halt is initiated, which could worsen the drawdown.

3. **Step 2 — Close All Positions:** `trading.close_all_positions(cancel_orders=True)` liquidates the entire portfolio.

4. **Step 3 — Locked Halt State:** A `self.halted` flag is set to `True`. The monitor loop continues to run but skips all polling and order submission for the remainder of the session. Manual intervention is required to reset.

5. **Step 4 — Emergency Alerts:** Email and SMS are dispatched concurrently via `asyncio.gather`. The email includes full context (drawdown %, opening equity, current equity). SMS is truncated to 1,600 characters for Twilio compatibility.

> **Halt Sequence:**
> `Trigger (−3% drawdown) → Cancel Orders → Close Positions → Set Halt Flag → Send Email + SMS`

#### Python Code Snippet

```python
import asyncio, smtplib, logging
from email.mime.text import MIMEText
from alpaca.trading.client import TradingClient
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

class PortfolioCircuitBreaker:
    DRAWDOWN_THRESHOLD = 0.03    # 3% intraday equity drop = HALT

    def __init__(self, trading_client: TradingClient,
                 smtp_cfg: dict, twilio_cfg: dict,
                 opening_equity: float):
        self.trading = trading_client
        self.smtp_cfg = smtp_cfg
        self.twilio_cfg = twilio_cfg
        self.opening_equity = opening_equity
        self.halted = False

    def _check_breach(self, current_equity: float) -> bool:
        drawdown = (self.opening_equity - current_equity) / self.opening_equity
        return drawdown >= self.DRAWDOWN_THRESHOLD

    async def _halt_sequence(self, current_equity: float):
        logger.critical("CIRCUIT BREAKER TRIGGERED — initiating halt sequence.")
        self.halted = True

        try:                                         # Step 1: Cancel orders
            self.trading.cancel_orders()
            logger.critical("Step 1 OK — All open orders cancelled.")
        except Exception as e:
            logger.error(f"Order cancel error: {e}")

        try:                                         # Step 2: Close positions
            self.trading.close_all_positions(cancel_orders=True)
            logger.critical("Step 2 OK — All positions closed.")
        except Exception as e:
            logger.error(f"Position close error: {e}")

        # Step 3: Enter locked halt state
        logger.critical("Step 3 OK — System locked. No further orders submitted.")

        # Step 4: Send alerts
        dd_pct = (self.opening_equity - current_equity) / self.opening_equity * 100
        msg = (f"TRADING HALT: Portfolio drew down {dd_pct:.2f}% intraday. "
               f"Opening: {self.opening_equity:,.2f}. Current: {current_equity:,.2f}. "
               f"All orders cancelled and positions closed.")
        await asyncio.gather(
            self._send_email(msg), self._send_sms(msg), return_exceptions=True
        )

    async def _send_email(self, message: str):
        msg = MIMEText(message)
        msg["Subject"] = "TRADING SYSTEM HALT — CIRCUIT BREAKER TRIGGERED"
        msg["From"] = self.smtp_cfg["from"]
        msg["To"]   = self.smtp_cfg["to"]
        try:
            with smtplib.SMTP_SSL(self.smtp_cfg["host"], 465) as server:
                server.login(self.smtp_cfg["user"], self.smtp_cfg["password"])
                server.send_message(msg)
            logger.info("Email alert sent.")
        except Exception as e:
            logger.error(f"Email send failure: {e}")

    async def _send_sms(self, message: str):
        try:
            tc = TwilioClient(self.twilio_cfg["sid"], self.twilio_cfg["token"])
            tc.messages.create(body=message[:1600],
                               from_=self.twilio_cfg["from_number"],
                               to=self.twilio_cfg["to_number"])
            logger.info("SMS alert sent.")
        except Exception as e:
            logger.error(f"SMS send failure: {e}")

    async def monitor(self, poll_interval: float = 10.0):
        """Run in background; polls account equity on interval."""
        while True:
            if self.halted:
                await asyncio.sleep(poll_interval)
                continue
            try:
                acct = self.trading.get_account()
                equity = float(acct.equity)
                if self._check_breach(equity):
                    await self._halt_sequence(equity)
            except Exception as e:
                logger.error(f"Circuit breaker poll error: {e}")
            await asyncio.sleep(poll_interval)
```

---

## Integration & Deployment Notes

**Dependency Summary**

`alpaca-py`, `pydantic >= 2.0`, `hmmlearn`, `pandas-ta`, `PyPortfolioOpt`, `transformers` (ProsusAI/finbert), `twilio`, `scikit-learn`, `scipy`, `numpy`, `multiprocessing` (stdlib).

**Process Supervision**

Each of the three pipeline processes (Section 1C) should be managed by a supervisor such as `systemd` or `supervisord`. If any process crashes, the supervisor restarts it independently without affecting the others. The circuit breaker (Section 4A) should run in the Execution process to ensure it has direct access to the trading client.

**Configuration Management**

All threshold values (`MAX_SPREAD_PCT`, `DRAWDOWN_THRESHOLD`, kelly fraction, VIX baselines, embargo sizes) should be externalized to a YAML or TOML configuration file loaded at startup. This allows parameter adjustment without code changes and enables environment-specific configs (paper vs. live).

**Logging & Observability**

All critical events (halt triggers, safety mode activation, deferred orders, rejected tickers) should emit structured JSON log lines to a centralized log aggregator (e.g., Datadog, CloudWatch). Set up alerting on any `CRITICAL` log level event independent of the SMS/email circuit breaker alerts.

> ⚠️ All code patterns provided in this document are reference implementations. They must be reviewed, tested in paper trading mode with real market data, and validated against your specific broker API version before any live deployment.
