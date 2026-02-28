import pytest
import numpy as np
from src.models.dcf import PositionSizer

@pytest.fixture
def config():
    return {
        "sizing": {
            "kelly_fraction": 0.25,
            "vix_baseline": 20.0,
            "vix_max": 40.0,
            "atr_baseline_pct": 0.015,
            "atr_max_pct": 0.04,
            "min_kelly_samples": 5,
            "max_kelly_cap": 0.15,
            "default_win_rate": 0.55,
            "default_win_loss": 1.5
        },
        "risk": {
            "adv_position_limit": 0.01
        }
    }

def test_kelly_default_fallback(config):
    sizer = PositionSizer(config)
    # Less than 5 samples
    f = sizer.calculate_kelly([0.05, -0.02])
    # Expected: (1.5 * 0.55 - 0.45) / 1.5 = (0.825 - 0.45) / 1.5 = 0.375 / 1.5 = 0.25
    # Scaled: 0.25 * 0.25 = 0.0625
    assert pytest.approx(f) == 0.0625

def test_kelly_rolling_stats(config):
    sizer = PositionSizer(config)
    # 6 samples: 4 wins (0.1), 2 losses (-0.05)
    # p = 4/6 = 0.666
    # avg_win = 0.1, avg_loss = 0.05 -> b = 2.0
    # kelly = (2 * 0.666 - 0.333) / 2 = 1.0 / 2 = 0.5
    # scaled = 0.5 * 0.25 = 0.125
    returns = [0.1, 0.1, 0.1, 0.1, -0.05, -0.05]
    f = sizer.calculate_kelly(returns)
    assert pytest.approx(f) == 0.125

def test_kelly_max_cap(config):
    sizer = PositionSizer(config)
    # High performance that would exceed cap
    # p = 0.9, b = 5.0 -> kelly = (5*0.9 - 0.1)/5 = 4.4/5 = 0.88
    # scaled = 0.88 * 0.25 = 0.22 -> Cap should pull it to 0.15
    returns = [0.5] * 10
    f = sizer.calculate_kelly(returns)
    assert f == 0.15
