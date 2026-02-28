import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("regime")

class KAMARegimeDetector:
    """
    Implements KAMA-MSR regime detection based on Piotr Pomorski's research.
    Uses an adaptive filter to distinguish between trending and sideways regimes.
    """
    def __init__(self, er_period: int = 10, fast_period: int = 2, slow_period: int = 30):
        self.er_period = er_period
        self.fast_sc = 2 / (fast_period + 1)
        self.slow_sc = 2 / (slow_period + 1)
        self.neutral_threshold = 0.3  # Efficiency Ratio below this is "Neutral"

    def calculate_kama(self, prices: np.ndarray) -> np.ndarray:
        """Calculates the Kaufman Adaptive Moving Average."""
        n = len(prices)
        kama = np.zeros(n)
        kama[self.er_period-1] = prices[self.er_period-1]

        for i in range(self.er_period, n):
            # 1. Efficiency Ratio (ER) = |Total Change| / Sum of Absolute Changes
            change = abs(prices[i] - prices[i - self.er_period])
            volatility = np.sum(np.abs(np.diff(prices[i - self.er_period : i + 1])))
            
            er = change / volatility if volatility != 0 else 0
            
            # 2. Smoothing Constant (SC)
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
            
            # 3. KAMA
            kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
            
        return kama

    def predict_state(self, prices: np.ndarray) -> str:
        """
        Classifies the market regime using KAMA slope and Efficiency Ratio (MSR proxy).
        """
        if len(prices) < self.er_period + 5:
            return "Neutral"

        # Calculate KAMA and ER for the latest window
        kama = self.calculate_kama(prices)
        
        # Latest Efficiency Ratio
        change = abs(prices[-1] - prices[-self.er_period])
        volatility = np.sum(np.abs(np.diff(prices[-self.er_period:])))
        er = change / volatility if volatility != 0 else 0
        
        # Slope of KAMA
        slope = (kama[-1] - kama[-3]) / kama[-3] if kama[-3] != 0 else 0
        
        # Logic: High ER means market is efficient (trending). Low ER means noise (MSR logic).
        if er < self.neutral_threshold:
            return "Neutral"
        
        if slope > 0.0001:  # Positive slope
            return "Bull"
        elif slope < -0.0001: # Negative slope
            return "Bear"
        
        return "Neutral"

class FastExitOverlay:
    """Detects sharp intraday SPY trend reversals to trigger early exits."""
    def __init__(self, config: dict):
        self.lookback = config["regime"]["fast_exit_lookback_bars"]
        self.threshold = config["regime"]["fast_exit_slope_threshold"]

    def calculate_slope(self, spy_closes: pd.Series) -> float:
        """Calculates normalized linear regression slope of recent prices."""
        if len(spy_closes) < self.lookback:
            return 0.0
        
        y = spy_closes.iloc[-self.lookback:].values
        x = np.arange(len(y))
        
        # Linear regression slope
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize by price to get rate of change %
        normalized_slope = slope / y.mean()
        return normalized_slope

    def should_fast_exit(self, current_regime: str, spy_closes: pd.Series) -> bool:
        """Checks if intraday trend diverges from bull regime."""
        if current_regime != "Bull":
            return False
            
        slope = self.calculate_slope(spy_closes)
        if slope < self.threshold:
            logger.warning(f"event=FAST_EXIT_TRIGGERED message='Regime is Bull, but SPY slope is {slope:.5f}'")
            return True
            
        return False
