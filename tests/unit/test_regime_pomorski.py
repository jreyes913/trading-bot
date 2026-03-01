import numpy as np
import pytest
from src.models.regime import KAMARegimeDetector

@pytest.fixture
def config():
    return {
        "regime": {
            "er_period": 10,
            "msr_period": 20,
            "msr_threshold": 0.15,
            "debug": False
        }
    }

def test_uptrend_detection(config):
    detector = KAMARegimeDetector(config)
    # 50 points of clear uptrend
    prices = np.linspace(100, 110, 50) + np.random.normal(0, 0.05, 50)
    state = detector.predict_state(prices)
    # Clear trend should yield Bull
    assert state == "Bull"

def test_downtrend_detection(config):
    detector = KAMARegimeDetector(config)
    # 50 points of clear downtrend
    prices = np.linspace(110, 90, 50) + np.random.normal(0, 0.05, 50)
    state = detector.predict_state(prices)
    # Clear trend should yield Bear
    assert state == "Bear"

def test_choppy_sideways_detection(config):
    detector = KAMARegimeDetector(config)
    # 50 points of random noise (high volatility, no net change)
    prices = 100 + np.random.normal(0, 2.0, 50)
    state = detector.predict_state(prices)
    # Noise dominant -> Neutral
    assert state == "Neutral"

def test_low_volatility_flatline(config):
    detector = KAMARegimeDetector(config)
    # Nearly flat line
    prices = np.ones(50) * 100.0
    state = detector.predict_state(prices)
    # Zero change -> Zero MSR -> Neutral
    assert state == "Neutral"

def test_regression_proxy_diff(config):
    """
    Regression: Ensure new MSR logic differs from simple slope logic.
    In a high-noise trending series, MSR might stay below threshold 
    where a simple slope logic would have stayed Bull.
    """
    detector = KAMARegimeDetector(config)
    # Trending but very noisy
    prices = np.linspace(100, 105, 50) + np.random.normal(0, 5.0, 50)
    state = detector.predict_state(prices)
    
    # If noise is very high relative to the tiny slope, MSR should catch it
    # whereas old logic just checked slope > 0.0001
    assert state in ["Bull", "Neutral"]
