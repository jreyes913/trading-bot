import numpy as np
import pandas as pd
import pytest
from src.models.regime import KAMARegimeDetector

@pytest.fixture
def config():
    return {
        "regime": {
            "er_period": 10,
            "msr_window": 40,
            "gamma": 0.15,
            "debug": False
        }
    }

def test_kama_calculation(config):
    detector = KAMARegimeDetector(config)
    # Perfectly trending path
    prices = np.arange(100, 150, dtype=float)
    kama = detector.calculate_kama(prices)
    assert len(kama) == len(prices)
    # KAMA should be increasing
    assert kama[-1] > kama[-10]

def test_kama_regime_transition(config):
    detector = KAMARegimeDetector(config)
    np.random.seed(42)
    
    # Generate synthetic bull market (Steady uptrend -> LV_Bull -> Returns Bear)
    t = np.linspace(0, 1, 200)
    bull_prices = 100 + 50 * t + np.random.normal(0, 0.01, 200)
    state = detector.predict_state(bull_prices)
    # Contrarion: Bull market returns Bear
    assert state in ["Bear", "Neutral"]

    # Generate synthetic bear market (Volatile downtrend -> HV_Bear -> Returns Bull)
    bear_prices = 150 - 50 * t + np.random.normal(0, 5.0, 200)
    state = detector.predict_state(bear_prices)
    # Contrarion: Bear market returns Bull
    assert state in ["Bull", "Neutral"]
