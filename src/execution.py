import os
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
from multiprocessing import Queue
from dotenv import load_dotenv
import yaml
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.models.gatekeeper import FundamentalGatekeeper
from src.models.dcf import DCFValuator, PositionSizer

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
            logging.FileHandler(f"{log_dir}/execution.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("execution")

logger = setup_logging()

class AlertManager:
    """Handles sending email and SMS alerts via SMTP gateway."""
    def __init__(self):
        self.host = os.getenv("SMTP_HOST")
        self.port = 465
        self.user = os.getenv("SMTP_USER")
        self.password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("SMTP_FROM")
        self.to_sms = os.getenv("SMTP_TO_SMS")

    def send_alert(self, subject: str, message: str):
        if not all([self.host, self.user, self.password, self.to_sms]):
            logger.error("Alert config missing in .env. Skipping alert.")
            return
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = self.to_sms
        try:
            with smtplib.SMTP_SSL(self.host, self.port) as server:
                server.login(self.user, self.password)
                server.send_message(msg)
            logger.info(f"event=ALERT_SENT message='Alert sent to {self.to_sms}'")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

class ExecutionEngine:
    def __init__(self, signal_queue: Queue, config: dict):
        self.signal_queue = signal_queue
        self.config = config
        
        # API Clients
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.trading = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Advanced Models
        self.gatekeeper = FundamentalGatekeeper()
        self.dcf = DCFValuator(config)
        self.sizer = PositionSizer(config)
        self.alerts = AlertManager()
        
        # State
        self.halted = False
        self.vix_symbol = "VIX" # Simplified: in practice we'd pull from FRED or a CBOE data source

    async def _get_latest_vol_data(self, symbol: str):
        """Retrieves current VIX and ticker-specific ATR%."""
        # For simulation, we return baseline values if data fetching fails
        vix = 20.0 
        atr_pct = 0.02
        adv_20 = 1_000_000
        
        try:
            # Calculate 20-day ADV and ATR
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=30)
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
            bars = self.data_client.get_stock_bars(req).df
            
            if not bars.empty:
                adv_20 = bars['volume'].tail(20).mean()
                # Simple ATR approximation
                atr_pct = (bars['high'] - bars['low']).tail(14).mean() / bars['close'].iloc[-1]
                
        except Exception as e:
            logger.error(f"Error fetching vol data for {symbol}: {e}")
            
        return vix, atr_pct, adv_20

    async def run(self):
        logger.info("Execution Engine started with DCF & Kelly sizing.")
        while True:
            try:
                loop = asyncio.get_event_loop()
                sig = await loop.run_in_executor(None, self.signal_queue.get)
                
                if sig["type"] == "FAST_EXIT" or sig["type"] == "SENTIMENT_EXIT":
                    logger.warning(f"event=EMERGENCY_EXIT type={sig['type']} symbol={sig['symbol']}")
                    self.trading.close_all_positions(cancel_orders=True)
                    continue

                if sig["type"] == "TRADE_SIGNAL" and sig["score"] >= 2:
                    symbol = sig["symbol"]
                    price = sig["price"]
                    
                    # 1. Fundamental Check (Simulation: we assume we have raw data)
                    # In production, we'd fetch this from FMP/EDGAR here.
                    raw_fundamentals = [{"ticker": symbol, "revenue": 1e9, "ebit": 1e8, "market_cap": 5e8, "total_debt": 1e8, "cash": 2e8, "shares_outstanding": 1e7}]
                    validated = self.gatekeeper.process(raw_fundamentals)
                    if not validated:
                        continue
                        
                    # 2. Monte Carlo DCF
                    est_value = self.dcf.estimate_value(ebitda=1.2e8, growth_mean=0.05, growth_std=0.02)
                    if not self.dcf.is_undervalued(price, est_value):
                        logger.info(f"event=TRADE_REJECTED symbol={symbol} reason='Not undervalued by DCF'")
                        continue
                        
                    # 3. Size the Position
                    vix, atr_pct, adv_20 = await self._get_latest_vol_data(symbol)
                    equity = float(self.trading.get_account().equity)
                    
                    f = self.sizer.calculate_kelly(win_rate=0.55, win_loss_ratio=1.5)
                    scaled_f = self.sizer.apply_vol_scaling(f, vix, atr_pct)
                    qty = self.sizer.final_shares(equity, price, scaled_f, adv_20)
                    
                    if qty > 0:
                        order = self.trading.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                        logger.info(f"event=ORDER_PLACED symbol={symbol} qty={qty} price={price}")
                        self.alerts.send_alert("TRADE PLACED", f"Bought {qty} shares of {symbol} at ${price:.2f}")

            except Exception as e:
                logger.error(f"Execution Engine error: {e}")

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    ExecutionEngine(Queue(), cfg).run()
