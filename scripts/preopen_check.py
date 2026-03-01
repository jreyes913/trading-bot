import os
import sys
import json
import yaml
import time
from dotenv import load_dotenv

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

def check_config():
    print("Checking config/config.yaml...")
    path = "config/config.yaml"
    if not os.path.exists(path):
        print(f"  FAILED: {path} not found.")
        return False
    try:
        with open(path, "r") as f:
            yaml.safe_load(f)
        print("  PASSED: config.yaml is valid YAML.")
        return True
    except Exception as e:
        print(f"  FAILED: Error parsing config.yaml: {e}")
        return False

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
        
        # Need to add src to path if not already there, but Makefile handles it
        try:
            from src.utils.symbols import is_valid_symbol
        except ImportError:
            # Fallback for direct execution
            sys.path.append(os.getcwd())
            from src.utils.symbols import is_valid_symbol

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
    print("=== PRE-OPEN READINESS CHECK ===")
    results = [
        check_env(),
        check_config(),
        check_cache(),
        check_universe()
    ]
    
    if all(results):
        print("\nSUCCESS: System is ready for market open.")
        sys.exit(0)
    else:
        print("\nFAILURE: System is NOT ready. Please fix errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
