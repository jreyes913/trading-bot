import os
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process
import time
import yaml
from dotenv import load_dotenv

# Import our modules
from src.ingestion import ResilientStreamManager
from src.indicators import IndicatorProcessor
from src.execution import ExecutionEngine

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/main.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("main")

logger = setup_logging()

def run_ingestion(bar_queue: Queue, config: dict):
    """Process 1: Data Ingestion (I/O bound)"""
    import asyncio
    logger.info("Starting Ingestion Process...")
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    # Ensure SPY is always included for regime detection
    symbols = list(set(config["trading_universe"] + ["SPY"]))
    
    manager = ResilientStreamManager(api_key, secret_key, symbols, bar_queue, config)
    asyncio.run(manager.run())

def run_indicators(bar_queue: Queue, signal_queue: Queue, config: dict):
    """Process 2: Indicator Calculation (CPU bound)"""
    logger.info("Starting Indicator Process...")
    processor = IndicatorProcessor(bar_queue, signal_queue, config)
    processor.run()

def run_execution(signal_queue: Queue, config: dict):
    """Process 3: Order Execution (Network I/O)"""
    import asyncio
    logger.info("Starting Execution Process...")
    executor = ExecutionEngine(signal_queue, config)
    asyncio.run(executor.run())

def main():
    logger.info("=== Alpaca Trading Bot Initializing ===")
    
    # 1. Load Config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Initialize Queues
    # We set maxsize to prevent memory bloat if a process lags
    bar_q = Queue(maxsize=2000)
    sig_q = Queue(maxsize=500)
    
    # 3. Define Processes
    processes = [
        Process(target=run_ingestion, args=(bar_q, config), name="Ingestion"),
        Process(target=run_indicators, args=(bar_q, sig_q, config), name="Indicators"),
        Process(target=run_execution, args=(sig_q, config), name="Execution")
    ]
    
    # 4. Launch Processes
    for p in processes:
        p.daemon = True # Ensure they die if main dies
        p.start()
        logger.info(f"Launched {p.name} process [PID: {p.pid}]")

    # 5. Monitor Processes
    try:
        while True:
            time.sleep(10)
            for p in processes:
                if not p.is_alive():
                    logger.critical(f"PROCESS CRASHED: {p.name}. Restarting system...")
                    # In a full production build, we might attempt to restart just the one process
                    # but for safety, we exit and let the supervisor (systemd) restart the whole bot.
                    return
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping bot...")
    finally:
        for p in processes:
            p.terminate()
            p.join()
        logger.info("All processes stopped. Cleanup complete.")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all platforms
    mp.set_start_method("spawn", force=True)
    main()
