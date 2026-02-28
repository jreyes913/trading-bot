import os
import json
import yaml
import logging
import yfinance as yf
from dotenv import load_dotenv

# Setup logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(f"{log_dir}/fundamental_cache.log"), logging.StreamHandler()]
    )
    return logging.getLogger("cache")

logger = setup_logging()
load_dotenv()

CACHE_FILE = "data/fundamental_cache.json"
CONFIG_FILE = "config/config.yaml"

def cache_fundamentals():
    logger.info("Starting Daily Fundamental Caching via yfinance...")
    
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    tickers = config.get("trading_universe", [])
    
    if not tickers:
        logger.error("No tickers found in config.")
        return

    cache_data = {}

    for ticker_sym in tickers:
        try:
            logger.info(f"Caching fundamentals for {ticker_sym}...")
            t = yf.Ticker(ticker_sym)
            info = t.info
            
            # Extract standard metrics for DCF and Gatekeeper
            cache_data[ticker_sym] = {
                "ticker": ticker_sym,
                "revenue": float(info.get("totalRevenue", 0)),
                "ebit": float(info.get("operatingCashflow", 0) * 0.8), # Conservative EBIT proxy
                "ebitda": float(info.get("ebitda", 0)),
                "total_debt": float(info.get("totalDebt", 0)),
                "cash": float(info.get("totalCash", 0)),
                "market_cap": float(info.get("marketCap", 0)),
                "shares_outstanding": float(info.get("sharesOutstanding", 0)),
                "growth_mean": 0.05, 
                "growth_std": 0.02
            }
        except Exception as e:
            logger.error(f"Failed to cache {ticker_sym}: {e}")

    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)
        
    logger.info(f"Successfully cached {len(cache_data)} tickers to {CACHE_FILE}")

if __name__ == "__main__":
    cache_fundamentals()
