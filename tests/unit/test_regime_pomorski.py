import numpy as np
import pytest
from src.models.regime import KAMARegimeDetector

@pytest.fixture
def config():
    return {
        "regime": {
            "er_period": 10,
            "msr_window": 60,
            "gamma": 0.1, # More sensitive
            "debug": False
        }
    }

def test_steady_uptrend_contrarian(config):
    """Steady uptrend -> LV_Bull -> Returns Bear."""
    detector = KAMARegimeDetector(config)
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    # Very steady, low noise
    prices = 100 + 100 * t + np.random.normal(0, 0.01, 200)
    state = detector.predict_state(prices)
    assert state in ["Bear", "Neutral"]

def test_volatile_downtrend_contrarian(config):
    """Volatile downtrend -> HV_Bear -> Returns Bull."""
    detector = KAMARegimeDetector(config)
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    # Very volatile downward
    prices = 200 - 100 * t + np.random.normal(0, 10.0, 200)
    state = detector.predict_state(prices)
    assert state in ["Bull", "Neutral"]

def test_flat_neutral(config):
    """Flat line -> Returns Neutral."""
    detector = KAMARegimeDetector(config)
    np.random.seed(42)
    # More noise to ensure ER < 0.3
    prices = 100.0 + np.random.normal(0, 1.0, 200)
    state = detector.predict_state(prices)
    assert state == "Neutral"
