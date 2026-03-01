import numpy as np
import pytest
from src.models.regime import KAMARegimeDetector

@pytest.fixture
def config():
    return {
        "regime": {
            "er_period": 10,
            "msr_window": 80,
            "gamma": 0.15,
            "debug": False
        }
    }

def test_steady_uptrend_contrarian(config):
    """Steady uptrend with noise -> LV_Bull -> Optimal Short (Returns Bear)."""
    config["regime"]["debug"] = True
    detector = KAMARegimeDetector(config)
    # 200 points of steady uptrend with moderate noise to avoid zero variance
    t = np.linspace(0, 1, 200)
    prices = 100 + 50 * t + np.random.normal(0, 0.5, 200)
    state = detector.predict_state(prices)
    # Market: LV_Bull (upwards trending, low vol) -> Returns Bear
    assert state == "Bear"

def test_volatile_downtrend_contrarian(config):
    """Volatile downtrend -> HV_Bear -> Optimal Long (Returns Bull)."""
    detector = KAMARegimeDetector(config)
    # 200 points of volatile downtrend
    t = np.linspace(0, 1, 200)
    prices = 150 - 50 * t + np.random.normal(0, 5.0, 200)
    state = detector.predict_state(prices)
    # Market: HV_Bear (downwards trending, high vol) -> Returns Bull
    assert state == "Bull"

def test_flat_neutral(config):
    """Flat line -> Other -> Returns Neutral."""
    detector = KAMARegimeDetector(config)
    prices = np.ones(200) * 100.0 + np.random.normal(0, 0.1, 200)
    state = detector.predict_state(prices)
    assert state == "Neutral"
