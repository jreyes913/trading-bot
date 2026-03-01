import json
import yaml
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.models.selector import InstitutionalSelector

# Setup logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(f"{log_dir}/universe_refresh.log"), logging.StreamHandler()]
    )
    return logging.getLogger("refresh")

logger = setup_logging()
load_dotenv()

LAST_CHECK_FILE = "data/last_universe_update.json"
CONFIG_FILE = "config/config.yaml"
TRADING_DAYS_WINDOW = 63

def should_refresh() -> bool:
    """Checks if 63 trading days (approx 90 calendar days) have passed."""
    if not os.path.exists(LAST_CHECK_FILE):
        return True
        
    with open(LAST_CHECK_FILE, "r") as f:
        data = json.load(f)
        last_date = datetime.fromisoformat(data["last_update"])
        
    # Using 90 calendar days as a proxy for 63 trading days
    return datetime.now() > (last_date + timedelta(days=90))

def run_refresh():
    logger.info("Starting Universe Preselection Phase...")
    
    # 1. Load full S&P 500 list from current config
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    # If trading_universe is already filtered, we might need the full list.
    # For now, we assume trading_universe contains the candidates.
    full_list = config.get("trading_universe", [])
    
    selector = InstitutionalSelector(config=config, target_count=40)
    
    # 2. Step 1: Fundamental Moat Filter
    logger.info("Step 1: Filtering for 'Moat' characteristics...")
    moat_df = selector.get_moat_candidates(full_list)
    
    if moat_df.empty:
        logger.error("No moat candidates found. Check criteria.")
        return

    # 3. Step 2: Calculate MRD, Beta, Sharpe
    logger.info("Step 2: Calculating Triple-Metrics (MRD, Beta, Sharpe)...")
    metrics_df = selector.calculate_triple_metrics(moat_df)
    
    # 4. Step 3: Final 40 selection
    new_40 = selector.select_best_40(metrics_df)
    
    # 5. Update Config
    config["trading_universe"] = new_40
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f)
        
    # 6. Save Last Update Date
    with open(LAST_CHECK_FILE, "w") as f:
        json.dump({"last_update": datetime.now().isoformat()}, f)
        
    logger.info(f"Successfully refreshed universe. New 40 tickers saved to {CONFIG_FILE}")

if __name__ == "__main__":
    if should_refresh():
        run_refresh()
    else:
        logger.info("Universe is up to date. No refresh needed.")
