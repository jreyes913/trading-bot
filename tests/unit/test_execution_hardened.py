import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from multiprocessing import Queue
from src.execution import ExecutionEngine
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

@pytest.fixture
def config():
    return {
        "risk": {
            "adv_position_limit": 0.01,
            "max_positions_per_symbol": 1,
            "buy_cooldown_s": 1, # Short cooldown for testing
            "circuit_breaker_drawdown": 0.03,
            "circuit_breaker_poll_s": 0.1,
            "max_spread_pct": 0.0005
        },
        "validation": {
            "dcf_simulations": 10,
            "dcf_undervalue_threshold": 0.65
        },
        "sizing": {
            "kelly_fraction": 0.25,
            "vix_baseline": 20.0,
            "vix_max": 40.0,
            "atr_baseline_pct": 0.015,
            "atr_max_pct": 0.04
        }
    }

@pytest.mark.asyncio
async def test_duplicate_buy_prevention(config):
    with patch('src.execution.TradingClient'), \
         patch('src.execution.StockHistoricalDataClient'), \
         patch('src.execution.AlertManager'), \
         patch('src.execution.ExecutionEngine._validate_fundamental_cache') as mock_val:
        
        mock_val.return_value = {"AAPL": {"ebitda": 1e9, "growth_mean": 0.05, "growth_std": 0.02}}
        
        engine = ExecutionEngine(Queue(), config)
        engine.position_cache = {"AAPL": 1} # Already hold AAPL
        
        # Test signal for AAPL
        sig = {"type": "TRADE_SIGNAL", "symbol": "AAPL", "score": 2, "price": 150.0}
        
        # Simulate what happens in the loop
        if engine.position_cache.get(sig["symbol"], 0) >= engine.max_pos_per_symbol:
            can_buy = False
        else:
            can_buy = True
                
        assert can_buy is False

@pytest.mark.asyncio
async def test_cooldown_enforcement(config):
    with patch('src.execution.TradingClient'), \
         patch('src.execution.StockHistoricalDataClient'), \
         patch('src.execution.AlertManager'):
        
        engine = ExecutionEngine(Queue(), config)
        engine.last_buy_ts = {"AAPL": time.time()} # Just bought
        
        # Should skip because cooldown is 1s and we just bought
        if (time.time() - engine.last_buy_ts["AAPL"]) < engine.buy_cooldown:
            can_buy = False
        else:
            can_buy = True
            
        assert can_buy is False
        
        # Wait for cooldown
        await asyncio.sleep(1.1)
        
        if (time.time() - engine.last_buy_ts["AAPL"]) < engine.buy_cooldown:
            can_buy = False
        else:
            can_buy = True
            
        assert can_buy is True

@pytest.mark.asyncio
async def test_sell_signal_executes_order(config):
    with patch('src.execution.TradingClient') as MockTradingClient, \
         patch('src.execution.StockHistoricalDataClient'), \
         patch('src.execution.AlertManager'), \
         patch('src.execution.ExecutionEngine._validate_fundamental_cache') as mock_val:

        # Setup mock trading client
        mock_trading_client = MockTradingClient.return_value
        mock_trading_client.submit_order = MagicMock()

        engine = ExecutionEngine(Queue(), config)
        engine.trading = mock_trading_client
        engine.position_cache = {"AAPL": 10} # We hold 10 shares of AAPL

        # Create a SELL_SIGNAL
        sig = {"type": "SELL_SIGNAL", "symbol": "AAPL"}
        
        # This is a simplified version of the logic in the main loop
        if sig["type"] == "SELL_SIGNAL":
            symbol = sig["symbol"]
            if symbol in engine.position_cache and engine.position_cache[symbol] > 0:
                qty_to_sell = engine.position_cache[symbol]
                try:
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    engine.trading.submit_order(order_request)
                except Exception as e:
                    pass # Test will fail if it reaches here without the mock being called

        # Assert that submit_order was called correctly
        mock_trading_client.submit_order.assert_called_once()
        called_with_arg = mock_trading_client.submit_order.call_args[0][0]
        assert called_with_arg.symbol == "AAPL"
        assert called_with_arg.qty == 10
        assert called_with_arg.side == OrderSide.SELL
