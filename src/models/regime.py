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

    def _calculate_er(self, prices: np.ndarray, index: int) -> float:
        """Shared helper for Efficiency Ratio calculation."""
        if index < self.er_period:
            return 0.0
        
        window = prices[index - self.er_period : index + 1]
        change = abs(window[-1] - window[0])
        volatility = np.sum(np.abs(np.diff(window)))
        
        return change / volatility if volatility != 0 else 0.0

    def calculate_kama(self, prices: np.ndarray) -> np.ndarray:
        """Calculates the Kaufman Adaptive Moving Average."""
        n = len(prices)
        kama = np.zeros(n)
        if n < self.er_period:
            return kama
            
        kama[self.er_period-1] = prices[self.er_period-1]

        for i in range(self.er_period, n):
            # 1. Efficiency Ratio (ER)
            er = self._calculate_er(prices, i)
            
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

        # Calculate KAMA for the whole window to ensure continuity
        kama = self.calculate_kama(prices)
        
        # Latest Efficiency Ratio using shared helper
        er = self._calculate_er(prices, len(prices) - 1)
        
        # Slope of KAMA (using latest 3 points for smoothing)
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
