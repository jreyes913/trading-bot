import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from multiprocessing import Queue
from datetime import datetime, timedelta
from src.execution import ExecutionEngine
from src.main import main
from alpaca.trading.enums import OrderSide

@pytest.fixture
def config():
    return {
        "risk": {
            "adv_position_limit": 0.01,
            "max_positions_per_symbol": 1,
            "buy_cooldown_s": 3600,
            "circuit_breaker_drawdown": 0.03,
            "circuit_breaker_poll_s": 0.1,
            "max_spread_pct": 0.0015
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
async def test_realized_return_reconstruction(config):
    with patch('src.execution.TradingClient'), \
         patch('src.execution.StockHistoricalDataClient'), \
         patch('src.execution.AlertManager'), \
         patch('src.execution.ExecutionEngine._validate_fundamental_cache'):
        
        engine = ExecutionEngine(Queue(), config)
        
        # Mock closed orders
        m1 = MagicMock(symbol="AAPL", filled_qty=10, filled_avg_price=100, side=OrderSide.BUY, filled_at=datetime.now(), created_at=datetime.now())
        m2 = MagicMock(symbol="AAPL", filled_qty=5, filled_avg_price=110, side=OrderSide.SELL, filled_at=datetime.now(), created_at=datetime.now())
        m3 = MagicMock(symbol="AAPL", filled_qty=5, filled_avg_price=105, side=OrderSide.SELL, filled_at=datetime.now(), created_at=datetime.now())
        m4 = MagicMock(symbol="MSFT", filled_qty=5, filled_avg_price=200, side=OrderSide.BUY, filled_at=datetime.now(), created_at=datetime.now())
        m5 = MagicMock(symbol="MSFT", filled_qty=5, filled_avg_price=190, side=OrderSide.SELL, filled_at=datetime.now(), created_at=datetime.now())
        
        engine.trading.get_orders.return_value = [m1, m2, m3, m4, m5]
        
        await engine._update_realized_returns()
        
        # Expected returns: [0.1, 0.05, -0.05]
        assert len(engine.realized_returns) == 3
        # Use approx for float comparisons
        assert any(abs(r - 0.1) < 1e-5 for r in engine.realized_returns)
        assert any(abs(r - 0.05) < 1e-5 for r in engine.realized_returns)
        assert any(abs(r - (-0.05)) < 1e-5 for r in engine.realized_returns)

@pytest.mark.asyncio
async def test_spread_filter_logic(config):
    with patch('src.execution.TradingClient'), \
         patch('src.execution.StockHistoricalDataClient'), \
         patch('src.execution.AlertManager'), \
         patch('src.execution.ExecutionEngine._validate_fundamental_cache'):
        
        engine = ExecutionEngine(Queue(), config)
        
        # Mock tight spread
        mock_quote = MagicMock(bid_price=100.0, ask_price=100.1) # 0.1% spread
        engine.data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
        
        passed = await engine._is_spread_tight("AAPL")
        assert passed is True
        assert engine.spread_pass_count == 1
        
        # Mock wide spread
        mock_quote_wide = MagicMock(bid_price=100.0, ask_price=101.0) # ~1.0% spread
        engine.data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote_wide}
        
        passed = await engine._is_spread_tight("AAPL")
        assert passed is False
        assert engine.spread_fail_count == 1

def test_main_lifecycle_alerts():
    # Test that main() sends start/stop alerts
    with patch('src.main.AlertManager') as mock_alert_cls, \
         patch('src.main.Process'), \
         patch('src.main.Queue'), \
         patch('src.main.open'), \
         patch('src.main.yaml.safe_load') as mock_load, \
         patch('src.main.time.sleep', side_effect=KeyboardInterrupt):
        
        mock_load.return_value = {"trading_universe": ["AAPL"]}
        mock_alerts = mock_alert_cls.return_value
        
        try:
            main()
        except KeyboardInterrupt:
            pass
            
        from unittest.mock import ANY
        # Verify BOT STARTED and BOT STOPPED were called
        mock_alerts.send_alert.assert_any_call("BOT STARTED", ANY)
        mock_alerts.send_alert.assert_any_call("BOT STOPPED", ANY)
