import numpy as np
import pytest
from src.models.regime import KAMARegimeDetector

@pytest.fixture
def config():
    return {
        "regime": {
            "er_period": 10,
            "msr_window": 80,
            "msr_regimes": 3,
            "debug": False
        }
    }

def test_uptrend_detection_robust(config):
    detector = KAMARegimeDetector(config)
    # 150 points of clear uptrend
    prices = np.linspace(100, 130, 150) + np.random.normal(0, 0.01, 150)
    state = detector.predict_state(prices)
    assert state == "Bull"

def test_downtrend_detection_robust(config):
    detector = KAMARegimeDetector(config)
    # 150 points of clear downtrend
    prices = np.linspace(130, 100, 150) + np.random.normal(0, 0.01, 150)
    state = detector.predict_state(prices)
    assert state == "Bear"

def test_choppy_neutral_robust(config):
    detector = KAMARegimeDetector(config)
    # Noisy sideways - should hit the ER < 0.2 filter
    prices = 100 + np.random.normal(0, 5.0, 150)
    state = detector.predict_state(prices)
    assert state == "Neutral"

def test_flat_line_neutral(config):
    detector = KAMARegimeDetector(config)
    prices = np.ones(150) * 100.0
    state = detector.predict_state(prices)
    assert state == "Neutral"
