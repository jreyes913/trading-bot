# 🛡️ How We Keep the Bot from Dying (Architecture)

Look, trading is risky. This doc explains how we make sure the bot doesn't do anything stupid and lose our money.

## 🧱 The Three Main Parts

| Part | What it's called | Why we care |
| :--- | :--- | :--- |
| **Connectivity** | Reconnect Loop | If the internet goes down, it keeps trying to get back online. It starts slow and gets faster, so we don't spam Alpaca. |
| **Vibe Check** | Heartbeat Monitor | If we don't hear from the market for 0.5 seconds, we assume the connection is dead and **kill all trades** immediately. Clutch. |
| **Validation** | Pydantic Gatekeeper | Before we trust any financial data (like from EDGAR), we check it. If the revenue is a negative number (which is impossible), we ignore it. |

## 🧮 Smart Money Math

| Concept | The Fancy Name | What it actually does |
| :--- | :--- | :--- |
| **Don't Overbet** | Fractional Kelly | We calculate the perfect bet size, then only use 25% of that. It keeps us from going broke after one bad trade. |
| **Volatility Scaling** | VIX Adjustment | If the market is getting super crazy (high VIX), we automatically make our trades smaller. |
| **Regime Detector** | HMM | A "brain" that checks if the market is Bull, Bear, or Neutral. We only buy when it's Bull. |
| **Fast Exit** | SPY Trend Slope | If the SPY (the whole market) starts crashing, we get out of our trades instantly, even if our other signals say everything's fine. |

## 🚀 Running Faster

| Issue | Fix | Why? |
| :--- | :--- | :--- |
| **Python is Slow** | Multi-Processing | We use 3 separate CPU processes. One for listening to data, one for doing math, and one for making trades. No lagging allowed. |
| **Lookahead Bias** | Purged Cross-Val | When we test the bot, we make sure it's not "cheating" by seeing future data during training. No Cap. |

## 🛑 Safety Breaks

| What | Trigger | What happens? |
| :--- | :--- | :--- |
| **Circuit Breaker** | -3% Intraday Loss | Bot cancels all orders and closes all positions. Stops trading for the day. |
| **Virtual Stop** | Price breach | A super fast monitor that sells as soon as a stop price is hit, even if the market is gapping. |
| **Bad Spread** | Bid-Ask > 0.05% | If the spread (the cost of trading) is too high, the bot just waits for it to get better. |
