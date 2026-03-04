import os
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from multiprocessing import Queue
from transformers import pipeline
import yaml

from src.models.regime import KAMARegimeDetector, FastExitOverlay

# Setup logging
def setup_logging(name="indicators"):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    fh = logging.FileHandler(f"{log_dir}/{name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

if __name__ == "__main__":
    logger = setup_logging()
else:
    logger = logging.getLogger("indicators")

class SentimentChangeDetector:
    """Tracks rolling sentiment score per ticker via FinBERT."""
    def __init__(self, config: dict):
        self.finbert = pipeline("sentiment-analysis", 
                                model="ProsusAI/finbert", 
                                truncation=True,
                                device=-1)
        self.history: dict[str, deque] = {}
        self.min_readings = config["sentiment"]["min_history_readings"]
        self.z_threshold = config["sentiment"]["zscore_exit_threshold"]

    def _get_score(self, text: str) -> float:
        try:
            result = self.finbert(text[:512])[0]
            label, conf = result["label"], result["score"]
            return conf if label == "positive" else (-conf if label == "negative" else 0.0)
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return 0.0

    def update(self, ticker: str, text: str) -> dict:
        score = self._get_score(text)
        if ticker not in self.history:
            self.history[ticker] = deque(maxlen=756)
        
        self.history[ticker].append(score)
        hist = np.array(self.history[ticker])

        if len(hist) < self.min_readings:
            return {"ticker": ticker, "score": score, "signal": "INSUFFICIENT_HISTORY"}

        mean_hist = np.mean(hist[:-1])
        std_hist = np.std(hist[:-1])
        z_score = (score - mean_hist) / std_hist if std_hist > 0 else 0.0
        signal = "EXIT_ALERT" if z_score < self.z_threshold else "HOLD"
        
        return {"ticker": ticker, "score": score, "z_score": z_score, "signal": signal}

class IndicatorProcessor:
    def __init__(self, bar_queue: Queue, signal_queue: Queue, config: dict):
        self.bar_queue = bar_queue
        self.signal_queue = signal_queue
        self.config = config
        # Store full bar dictionaries instead of just close prices
        self.buffers = defaultdict(lambda: deque(maxlen=500))
        
        # Advanced Models
        self.sentiment_detector = SentimentChangeDetector(config)
        self.regime_detector = KAMARegimeDetector(config)
        self.fast_exit = FastExitOverlay(config)
        
        self.current_regime = "Neutral"
        self.spy_symbol = "SPY"

    def run(self):
        logger.info("Indicator processor started with KAMA-MSR Regime Detection.")
        from finta import TA
        
        while True:
            try:
                item = self.bar_queue.get()
                
                # 1. Handle News Sentiment
                if "text" in item:
                    res = self.sentiment_detector.update(item["symbol"], item["text"])
                    if res["signal"] == "EXIT_ALERT":
                        self.signal_queue.put({"symbol": item["symbol"], "type": "SENTIMENT_EXIT", "data": res})
                    continue

                # 2. Update Buffers with full bar data
                symbol = item["symbol"]
                self.buffers[symbol].append(item)
                
                if self.config.get("regime", {}).get("debug", False):
                    logger.debug(f"Processing bar for {symbol} (buffer size: {len(self.buffers[symbol])})")

                # 3. Special Handling for SPY (Global Regime)
                if symbol == self.spy_symbol:
                    if len(self.buffers[self.spy_symbol]) > 30:
                        # Extract closes for regime detector
                        spy_closes = np.array([b["close"] for b in self.buffers[self.spy_symbol]])
                        new_regime = self.regime_detector.predict_state(spy_closes)
                        if new_regime != self.current_regime:
                            logger.info(f"event=REGIME_CHANGED old={self.current_regime} new={new_regime}")
                            self.current_regime = new_regime
                        
                        # Check for Fast Exit reversal
                        if self.fast_exit.should_fast_exit(self.current_regime, pd.Series(spy_closes)):
                            self.signal_queue.put({"type": "FAST_EXIT", "symbol": "GLOBAL"})

                # 4. Standard Signal Generation
                if len(self.buffers[symbol]) < 30:
                    continue

                # Technical Calculations using full OHLC data
                df = pd.DataFrame(list(self.buffers[symbol]))
                # Ensure column names are what finta expects
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                rsi_val = TA.RSI(df).iloc[-1]
                macd_df = TA.MACD(df)
                macd_val = macd_df['MACD'].iloc[-1]
                
                # Scoring (Strict: Only allow Buy if Regime is Bull)
                score = 0
                if self.current_regime == "Bull":
                    if macd_val > 0 and rsi_val < 70: 
                        score += 2
                    # Sell signal if conditions are no longer met
                    elif macd_val <= 0 or rsi_val >= 70:
                        self.signal_queue.put({
                            "type": "SELL_SIGNAL",
                            "symbol": symbol,
                            "reason": f"MACD_RSI_EXIT: macd={macd_val:.2f}, rsi={rsi_val:.2f}"
                        })

                elif self.current_regime == "Bear":
                    # Exit any open position if regime turns bearish
                    self.signal_queue.put({
                        "type": "SELL_SIGNAL",
                        "symbol": symbol,
                        "reason": "BEAR_REGIME_EXIT"
                    })
                
                if score > 0:
                    self.signal_queue.put({
                        "type": "TRADE_SIGNAL",
                        "symbol": symbol,
                        "score": score,
                        "price": item["close"],
                        "regime": self.current_regime,
                        "timestamp": item["timestamp"]
                    })

            except Exception as e:
                logger.error(f"Error in indicator loop: {e}")

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    IndicatorProcessor(Queue(), Queue(), cfg).run()
