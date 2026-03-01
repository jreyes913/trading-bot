import os
import asyncio
import logging
import json
import time
from multiprocessing import Queue
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yaml
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.models.gatekeeper import FundamentalGatekeeper
from src.models.dcf import DCFValuator, PositionSizer
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
            logging.FileHandler(f"{log_dir}/execution.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("execution")

logger = setup_logging()

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
        
        # Risk & State
        self.halted = False
        self.opening_equity = None
        self.position_cache = {} # symbol -> qty
        self.last_buy_ts = {}    # symbol -> timestamp
        self.realized_returns = [] # list of floats for Kelly
        
        # Observability counters
        self.spread_pass_count = 0
        self.spread_fail_count = 0
        
        # Constants from config
        self.max_pos_per_symbol = config["risk"].get("max_positions_per_symbol", 1)
        self.buy_cooldown = config["risk"].get("buy_cooldown_s", 3600)
        self.drawdown_limit = config["risk"]["circuit_breaker_drawdown"]
        self.poll_interval = config["risk"]["circuit_breaker_poll_s"]
        self.max_spread = config["risk"]["max_spread_pct"]

    async def _update_position_cache(self):
        """Syncs local position state with Alpaca."""
        try:
            positions = self.trading.get_all_positions()
            self.position_cache = {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Failed to update position cache: {e}")

    async def _check_circuit_breaker(self):
        """Monitors intraday drawdown and triggers halt."""
        try:
            acct = self.trading.get_account()
            equity = float(acct.equity)
            
            if self.opening_equity is None:
                self.opening_equity = equity
                logger.info(f"Session opening equity captured: ${self.opening_equity:,.2f}")
                return

            drawdown = (self.opening_equity - equity) / self.opening_equity
            if drawdown >= self.drawdown_limit:
                logger.critical(f"event=CIRCUIT_BREAKER_TRIGGERED drawdown={drawdown:.2%} opening_equity={self.opening_equity} current_equity={equity}")
                self.halted = True
                self.trading.close_all_positions(cancel_orders=True)
                self.alerts.send_alert("TRADING HALT", f"Bot halted. Drawdown: {drawdown:.2%}")
        except Exception as e:
            logger.error(f"Circuit breaker check failed: {e}")

    async def _is_spread_tight(self, symbol: str) -> bool:
        """Enforces spread filter before entry."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(req)[symbol]
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid
            
            if spread_pct > self.max_spread:
                self.spread_fail_count += 1
                logger.warning(f"event=SPREAD_FILTER_FAIL symbol={symbol} spread={spread_pct:.5%} limit={self.max_spread:.5%} fail_count={self.spread_fail_count}")
                return False
            
            self.spread_pass_count += 1
            logger.info(f"event=SPREAD_FILTER_PASS symbol={symbol} spread={spread_pct:.5%} pass_count={self.spread_pass_count}")
            return True
        except Exception as e:
            logger.error(f"Spread check failed for {symbol}: {e}")
            return False

    def _validate_fundamental_cache(self, cache_path: str):
        """Validates cache schema and freshness at startup."""
        if not os.path.exists(cache_path):
            logger.critical("event=STARTUP_FAILED reason='Fundamental cache missing'")
            raise FileNotFoundError(f"Cache missing at {cache_path}")
            
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) > 86400:
            logger.warning("event=STALE_CACHE_DETECTED message='Cache older than 24h'")

        with open(cache_path, "r") as f:
            cache = json.load(f)
            
        required_fields = ["ebitda", "growth_mean", "growth_std"]
        valid_count = 0
        for sym, data in cache.items():
            if all(k in data for k in required_fields):
                valid_count += 1
        
        logger.info(f"event=CACHE_VALIDATED valid_symbols={valid_count} total_symbols={len(cache)}")
        return cache

    async def _get_latest_vol_data(self, symbol: str):
        vix = 20.0 
        atr_pct = 0.02
        adv_20 = 1_000_000
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
            bars = self.data_client.get_stock_bars(req).df
            if not bars.empty:
                adv_20 = bars['volume'].tail(20).mean()
                atr_pct = (bars['high'] - bars['low']).tail(14).mean() / bars['close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching vol data for {symbol}: {e}")
        return vix, atr_pct, adv_20

    async def _update_realized_returns(self):
        """Polls closed orders to calculate realized returns for Kelly sizing."""
        try:
            # Fetch closed orders from the last 7 days
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                until=datetime.now(),
                after=datetime.now() - timedelta(days=7)
            )
            closed_orders = self.trading.get_orders(req)
            
            # Group by symbol to pair buys/sells
            by_symbol = {}
            for o in closed_orders:
                if o.filled_avg_price and o.filled_qty:
                    sym = o.symbol
                    if sym not in by_symbol: by_symbol[sym] = []
                    by_symbol[sym].append(o)
            
            new_returns = []
            for sym, orders in by_symbol.items():
                # Sort by filled_at
                orders.sort(key=lambda x: x.filled_at)
                
                # Simple FIFO pairing for return reconstruction
                buys = [o for o in orders if o.side == OrderSide.BUY]
                sells = [o for o in orders if o.side == OrderSide.SELL]
                
                while buys and sells:
                    b = buys.pop(0)
                    s = sells.pop(0)
                    
                    # Compute % return
                    ret = (float(s.filled_avg_price) - float(b.filled_avg_price)) / float(b.filled_avg_price)
                    new_returns.append(ret)
            
            if new_returns:
                self.realized_returns = new_returns[-100:] # Keep last 100
                logger.info(f"event=REALIZED_RETURNS_UPDATED count={len(self.realized_returns)} latest_mean={np.mean(self.realized_returns):.4f}")
                
        except Exception as e:
            logger.error(f"Failed to update realized returns: {e}")

    async def monitor_risk(self):
        """Background task for risk guardrails."""
        while True:
            await self._check_circuit_breaker()
            await self._update_position_cache()
            await self._update_realized_returns()
            await asyncio.sleep(self.poll_interval)

    async def run(self):
        logger.info("Execution Engine started with hardened risk controls.")
        
        # 1. Startup Validation
        fund_cache = self._validate_fundamental_cache("data/fundamental_cache.json")
        
        # 2. Launch risk monitor
        asyncio.create_task(self.monitor_risk())

        while True:
            try:
                if self.halted:
                    await asyncio.sleep(1)
                    continue

                loop = asyncio.get_event_loop()
                sig = await loop.run_in_executor(None, self.signal_queue.get)
                
                if sig["type"] in ["FAST_EXIT", "SENTIMENT_EXIT"]:
                    logger.warning(f"event=EMERGENCY_EXIT type={sig['type']} symbol={sig['symbol']}")
                    self.trading.close_all_positions(cancel_orders=True)
                    continue

                if sig["type"] == "TRADE_SIGNAL" and sig["score"] >= 2:
                    symbol = sig["symbol"]
                    price = sig["price"]
                    
                    # Hardening: Duplicate Buy Prevention
                    if self.position_cache.get(symbol, 0) >= self.max_pos_per_symbol:
                        logger.info(f"event=ORDER_SKIPPED symbol={symbol} reason='Max positions reached' current_qty={self.position_cache.get(symbol)}")
                        continue
                        
                    # Hardening: Cooldown Enforcement
                    if symbol in self.last_buy_ts:
                        if (time.time() - self.last_buy_ts[symbol]) < self.buy_cooldown:
                            logger.info(f"event=ORDER_SKIPPED symbol={symbol} reason='Buy cooldown active' remaining={int(self.buy_cooldown - (time.time() - self.last_buy_ts[symbol]))}s")
                            continue

                    # Hardening: Spread Filter
                    if not await self._is_spread_tight(symbol):
                        continue

                    # Fundamental Check from Validated Cache
                    if symbol not in fund_cache:
                        logger.warning(f"event=ORDER_SKIPPED symbol={symbol} reason='No fundamental data in cache'")
                        continue
                        
                    raw_data = fund_cache[symbol]
                    validated = self.gatekeeper.process([raw_data])
                    if not validated:
                        continue
                        
                    est_value = self.dcf.estimate_value(
                        ebitda=raw_data["ebitda"], 
                        growth_mean=raw_data["growth_mean"], 
                        growth_std=raw_data["growth_std"]
                    )
                    
                    if not self.dcf.is_undervalued(price, est_value):
                        logger.info(f"event=ORDER_SKIPPED symbol={symbol} reason='Not undervalued by DCF' price={price} est={est_value:.2f}")
                        continue
                        
                    # Size the Position using Rolling Kelly
                    vix, atr_pct, adv_20 = await self._get_latest_vol_data(symbol)
                    equity = float(self.trading.get_account().equity)
                    
                    f = self.sizer.calculate_kelly(realized_trades=self.realized_returns)
                    scaled_f = self.sizer.apply_vol_scaling(f, vix, atr_pct)
                    qty = self.sizer.final_shares(equity, price, scaled_f, adv_20)
                    
                    if qty > 0:
                        try:
                            order = self.trading.submit_order(MarketOrderRequest(
                                symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                            ))
                            self.last_buy_ts[symbol] = time.time()
                            logger.info(f"event=ORDER_PLACED symbol={symbol} qty={qty} price={price} kelly_f={f:.4f} scaled_f={scaled_f:.4f}")
                            self.alerts.send_alert("TRADE PLACED", f"Bought {qty} shares of {symbol} at ${price:.2f}")
                        except Exception as e:
                            logger.error(f"Order submission failed: {e}")

            except Exception as e:
                logger.error(f"Execution Engine main loop error: {e}")

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    engine = ExecutionEngine(Queue(), cfg)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
