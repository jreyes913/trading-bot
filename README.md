# 📈 My Alpaca Trading Bot (No Cap Edition)

This is a super solid bot for trading on Alpaca. I built it to be "anti-fragile," which basically means it won't just crash and burn when the market gets weird. It’s got a 20-step plan to make sure we're making smart moves and not losing the bag.

## 🚀 Cool Stuff This Bot Does

| Feature | What's the point? |
| :--- | :--- |
| **Internet Safety Net** | If the internet glitches for even half a second, it kills all trades so we don't get stuck in a bad spot. |
| **Triple Brain Power** | It uses 3 separate processes so it can read data, crunch numbers, and trade at the same time without lagging. |
| **Emergency Brake** | If the portfolio drops 3% in one day, it shuts down everything. FR, safety first. |
| **Don't Bet the Farm** | It uses "Kelly" math to make sure we only bet a small slice of our money at a time. |
| **Vibe Check** | It literally reads the news and uses AI to see if the "vibes" are off for a stock before buying. |

## 🧠 How the Bot Thinks

| Step | Brain Part | What it's doing |
| :--- | :--- | :--- |
| **1** | **The Listener** | Just hangs out on the internet waiting for stock prices to move. |
| **2** | **The Math Nerd** | Calculates things like RSI and MACD (basically checking if a stock is "cheap" or "expensive"). |
| **3** | **The Executor** | Pulls the trigger and buys/sells if the math looks good and the spread isn't trash. |

## 🛠️ Getting Started

### Prerequisites
- Python 3.11+
- An Alpaca account (Paper trading is fine, don't lose real money yet).

### Setting Up the Vibes
I'm using a specific folder for all my finance projects. Make sure you use it too:
```bash
source /home/jose/venvs/finance/bin/activate
```

### Installation
1. Grab the code:
   ```bash
   git clone https://github.com/jreyes913/trading-bot.git
   cd trading-bot
   ```
2. Install the heavy hitters (TA-Lib, VectorBT, etc.):
   ```bash
   pip install -r requirements.txt
   ```
3. Hide your keys:
   ```bash
   cp .env.example .env
   # Open .env and paste your Alpaca keys in there.
   ```

## ⚙️ Changing Settings
If you want to change how much the bot bets or when the emergency brake kicks in, check out `config/config.yaml`. It's all in there.

## 📜 Legal Stuff
Check the [LICENSE](LICENSE) file, but basically, don't sue me if the bot takes an L.
