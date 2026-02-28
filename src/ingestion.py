import asyncio
import time
import logging
import os
from collections import deque
from multiprocessing import Queue
from dotenv import load_dotenv
import yaml

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # In a real scenario, we would use config/logging.yaml
    # For now, we setup a basic structured-like logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/ingestion.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ingestion")

logger = setup_logging()

class HeartbeatMonitor:
    """Tracks last message timestamp; fires alert if threshold exceeded."""
    def __init__(self, threshold_ms: float = 500.0):
        self.threshold_ms = threshold_ms
        self.last_tick = time.monotonic()

    def pulse(self):
        self.last_tick = time.monotonic()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_tick) * 1000 > self.threshold_ms

class ResilientStreamManager:
    def __init__(self, api_key: str, secret_key: str, symbols: list, bar_queue: Queue, config: dict):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols
        self.bar_queue = bar_queue
        
        # Config values
        self.base_backoff = config["connectivity"]["ws_base_backoff_s"]
        self.max_backoff = config["connectivity"]["ws_max_backoff_s"]
        self.hb_threshold = config["connectivity"]["heartbeat_threshold_ms"]
        
        self.hb = HeartbeatMonitor(threshold_ms=self.hb_threshold)
        self.trading = TradingClient(api_key, secret_key, paper=True)
        
        self._safety_mode = False
        self._attempt = 0

    async def _on_bar(self, bar):
        """Callback for every incoming bar."""
        self.hb.pulse()
        
        # Convert bar object to dict for the queue
        bar_data = {
            "symbol": bar.symbol,
            "close": bar.close,
            "high": bar.high,
            "low": bar.low,
            "open": bar.open,
            "volume": bar.volume,
            "timestamp": bar.timestamp.isoformat()
        }
        
        try:
            self.bar_queue.put_nowait(bar_data)
        except Exception as e:
            logger.error(f"Error putting bar into queue: {e}")

        if self._safety_mode:
            logger.info(f"event=SAFETY_MODE_CLEARED message='Bar received for {bar.symbol}; exiting safety mode.'")
            self._safety_mode = False

    async def _watchdog(self):
        """Continuously checks heartbeat; triggers safety mode on lag."""
        while True:
            await asyncio.sleep(0.1)  # check every 100ms
            if self.hb.is_stale() and not self._safety_mode:
                logger.critical(f"event=WS_HEARTBEAT_BREACH message='STREAM LAG > {self.hb_threshold}ms — entering Safety Mode'")
                self._safety_mode = True
                await self._flatten_all_positions()

    async def _flatten_all_positions(self):
        """Emergency flatten: cancel orders then close all positions."""
        try:
            # Step 1: Cancel orders
            self.trading.cancel_orders()
            logger.critical("event=ORDERS_CANCELLED message='Step 1 OK — All open orders cancelled.'")
            
            # Step 2: Close positions
            self.trading.close_all_positions(cancel_orders=True)
            logger.critical("event=POSITIONS_CLOSED message='Step 2 OK — All positions closed.'")
            
            logger.critical("event=SAFETY_MODE_ACTIVATED message='System entered safety mode; positions flatten initiated.'")
        except Exception as e:
            logger.error(f"Flatten error: {e}")

    async def _run_stream(self):
        """Main stream loop with exponential backoff."""
        backoff = self.base_backoff
        while True:
            try:
                self._attempt += 1
                logger.info(f"event=WS_RECONNECT_ATTEMPT message='WS connect attempt #{self._attempt}' backoff_s={backoff}")
                
                wss = StockDataStream(self.api_key, self.secret_key)
                wss.subscribe_bars(self._on_bar, *self.symbols)
                
                # Start the stream
                await wss._run_forever()
                
                # If it exits cleanly, reset backoff
                backoff = self.base_backoff
                self._attempt = 0
                logger.info("event=WS_RECONNECT_SUCCESS message='WebSocket disconnected cleanly; resetting backoff.'")
            
            except Exception as e:
                logger.error(f"WS error: {e}. Retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

    async def run(self):
        """Starts the watchdog and the stream manager."""
        await asyncio.gather(
            self._watchdog(),
            self._run_stream()
        )

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # This block is for testing ingestion standalone
    try:
        config = load_config()
        symbols = config["trading_universe"][:10]  # Just watch first 10 for testing
        bar_q = Queue(maxsize=1000)
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        manager = ResilientStreamManager(api_key, secret_key, symbols, bar_q, config)
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        logger.info("Ingestion process stopped by user.")
    except Exception as e:
        logger.error(f"Ingestion process failed: {e}")
