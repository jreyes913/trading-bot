# 📋 The Bot Rules (Management Blueprint)

This doc is for the boring (but super important) management stuff. Where do things go? How do we fix them if they break?

## 📁 Where is Everything?

| Folder | What goes there? | Why? |
| :--- | :--- | :--- |
| `src/` | The actual code (the brains). | Keep all the logic in one place. |
| `config/` | Settings like how much risk we can take. | Don't hard-code thresholds, it makes testing easier. |
| `data/` | Where the bot saves its "homework" (like company info). | We don't want to re-download everything from EDGAR every day. |
| `logs/` | Where the bot records everything it does. | If something breaks, we check here. |
| `.env` | **SUPER SECRET KEYS.** | Never show anyone. This is your Alpaca password. |

## 🤫 Secrets & Settings

| Where | What's in it? | Important Rule |
| :--- | :--- | :--- |
| **.env** | Alpaca Keys, Twilio Keys, SMTP. | **DO NOT COMMIT THIS TO GITHUB.** |
| **.env.example** | A template for your keys. | Commit this, but leave it empty. |
| **config/config.yaml** | Risk levels, bet sizes, thresholds. | Feel free to change this as you learn. |

## 🛠️ What to Do if Things Break (SOPs)

| Situation | Action |
| :--- | :--- |
| **Internet Lag > 0.5s** | Bot goes into "Safety Mode." It will kill all trades. Just wait for the internet to come back, the bot will auto-reconnect. |
| **Loss > 3% in a Day** | Bot shuts down. It will send you a text. **Do NOT restart it** until the next day. Take the L, learn, and try again tomorrow. |
| **Data Rejected** | Check `data/rejected_tickers/`. If the bot is ignoring a ticker you like, it's because the data was trash or weird. |

## 📝 Keeping Tabs (Logging)

| Log Type | What is it for? |
| :--- | :--- |
| **trading.jsonl** | Everything the bot does. |
| **critical.jsonl** | Only the super serious stuff (like a crash or a trade-kill). |

## 🏗️ Makefile Commands

| Command | What it does |
| :--- | :--- |
| `make run` | Starts the bot. |
| `make test` | Runs the automated tests. |
| `make lint` | Checks your code for messy stuff. |
| `make halt` | Forces an emergency shutdown. |
