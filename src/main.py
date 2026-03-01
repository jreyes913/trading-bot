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
from src.alerts import AlertManager

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
    alerts = AlertManager()
    
    # 1. Load Config
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        alerts.send_alert("BOT STARTUP FAILED", f"Error loading config: {e}")
        return

    alerts.send_alert("BOT STARTED", f"Trading bot is now online with {len(config.get('trading_universe', []))} symbols.")
    
    # 2. Initialize Queues
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
        p.daemon = True
        p.start()
        logger.info(f"Launched {p.name} process [PID: {p.pid}]")

    stop_reason = "Unknown"
    try:
        while True:
            time.sleep(10)
            for p in processes:
                if not p.is_alive():
                    stop_reason = f"Process {p.name} CRASHED"
                    logger.critical(f"PROCESS CRASHED: {p.name}. Restarting system...")
                    return
    except KeyboardInterrupt:
        stop_reason = "Manual shutdown (Ctrl+C)"
        logger.info("Shutdown signal received. Stopping bot...")
    except Exception as e:
        stop_reason = f"Fatal error: {e}"
        logger.error(f"Unexpected error in main: {e}")
    finally:
        alerts.send_alert("BOT STOPPED", f"Trading bot has gone offline. Reason: {stop_reason}")
        for p in processes:
            p.terminate()
            p.join()
        logger.info("All processes stopped. Cleanup complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
