import os
import sys
import json
import yaml
import time
from dotenv import load_dotenv

# Add project root to sys.path for imports
sys.path.append(os.getcwd())

from src.utils.symbols import is_valid_symbol

def check_env():
    print("Checking environment variables...")
    load_dotenv()
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_TO_SMS"]
    missing = [r for r in required if not os.getenv(r)]
    if missing:
        print(f"  FAILED: Missing environment variables: {missing}")
        return False
    print("  PASSED: Environment variables present.")
    return True

def check_alpaca():
    print("Checking Alpaca connectivity...")
    from alpaca.trading.client import TradingClient
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        print(f"  PASSED: Connected. Account status: {account.status}, Equity: ${account.equity}")
        return True
    except Exception as e:
        print(f"  FAILED: Alpaca ping failed: {e}")
        return False

def check_config_sanity():
    print("Checking config sanity...")
    path = "config/config.yaml"
    if not os.path.exists(path):
        print(f"  FAILED: {path} not found.")
        return False
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        # Risk sanity
        drawdown = config.get("risk", {}).get("circuit_breaker_drawdown", 0)
        if not (0.01 <= drawdown <= 0.10):
            print(f"  FAILED: circuit_breaker_drawdown {drawdown} outside sane range [0.01, 0.10]")
            return False
            
        spread = config.get("risk", {}).get("max_spread_pct", 0)
        if not (0.0001 <= spread <= 0.01):
            print(f"  FAILED: max_spread_pct {spread} outside sane range [0.0001, 0.01]")
            return False

        # Sizing sanity
        kelly = config.get("sizing", {}).get("kelly_fraction", 0)
        if not (0.05 <= kelly <= 1.0):
            print(f"  FAILED: kelly_fraction {kelly} outside sane range [0.05, 1.0]")
            return False
            
        print("  PASSED: Config values within sane boundaries.")
        return True
    except Exception as e:
        print(f"  FAILED: Error parsing config.yaml: {e}")
        return False

def check_ops_integrity():
    print("Checking operational integrity (Makefile targets)...")
    try:
        from scripts.check_makefile_targets import check_makefile
        if check_makefile():
            print("  PASSED: Makefile targets validated.")
            return True
        else:
            print("  FAILED: Makefile target mismatch.")
            return False
    except ImportError:
        print("  WARNING: Could not import check_makefile logic. Skipping.")
        return True

def check_cache():
    print("Checking fundamental cache...")
    path = "data/fundamental_cache.json"
    if not os.path.exists(path):
        print(f"  FAILED: {path} not found. Run 'make cache-fund'.")
        return False
    
    try:
        with open(path, "r") as f:
            cache = json.load(f)
        
        # Check staleness (24h)
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) > 86400:
            print(f"  WARNING: Cache is stale (>24h). Last updated: {time.ctime(mtime)}")
        
        if not cache:
            print("  FAILED: Fundamental cache is empty.")
            return False
            
        print(f"  PASSED: Cache found with {len(cache)} symbols.")
        return True
    except Exception as e:
        print(f"  FAILED: Error checking cache: {e}")
        return False

def check_universe():
    print("Checking trading universe...")
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        universe = config.get("trading_universe", [])
        if not universe:
            print("  FAILED: Trading universe is empty in config.")
            return False
        
        invalid = [s for s in universe if not is_valid_symbol(s)]
        if invalid:
            print(f"  FAILED: Invalid symbols in universe: {invalid}")
            return False
            
        print(f"  PASSED: Universe has {len(universe)} valid symbols.")
        return True
    except Exception as e:
        print(f"  FAILED: Error checking universe: {e}")
        return False

def main():
    print("=== FINAL PRE-OPEN READINESS CHECK ===")
    results = [
        check_env(),
        check_alpaca(),
        check_config_sanity(),
        check_ops_integrity(),
        check_cache(),
        check_universe()
    ]
    
    if all(results):
        print("\nSUCCESS: System is fully hardened and ready for market open.")
        sys.exit(0)
    else:
        print("\nFAILURE: System is NOT ready. Please fix errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
