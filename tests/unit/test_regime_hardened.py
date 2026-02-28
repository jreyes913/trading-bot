import numpy as np
import pandas as pd
import pytest
from src.models.regime import KAMARegimeDetector

def test_er_calculation():
    detector = KAMARegimeDetector(er_period=10)
    # Perfectly trending path
    prices = np.arange(20, dtype=float)
    er = detector._calculate_er(prices, 15)
    # Total change = 10, total volatility = 10, ER should be 1.0
    assert pytest.approx(er) == 1.0

    # Choppy path
    choppy_prices = np.array([10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 11])
    er_choppy = detector._calculate_er(choppy_prices, 10)
    # Change = 1, Volatility = 13, ER should be 1/13
    assert pytest.approx(er_choppy) == 1/13

def test_kama_regime_transition():
    detector = KAMARegimeDetector(er_period=10)
    
    # Generate synthetic bull market
    bull_prices = np.linspace(100, 110, 50)
    state_bull = detector.predict_state(bull_prices)
    assert state_bull == "Bull"

    # Generate synthetic bear market
    bear_prices = np.linspace(110, 90, 50)
    state_bear = detector.predict_state(bear_prices)
    assert state_bear == "Bear"

    # Generate flat/neutral market
    flat_prices = np.ones(50) * 100 + np.random.normal(0, 0.1, 50)
    state_neutral = detector.predict_state(flat_prices)
    assert state_neutral == "Neutral"
