import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("regime")

class KAMARegimeDetector:
    """
    Implements KAMA-MSR regime detection based on Piotr Pomorski's research.
    
    The model uses the Kaufman Adaptive Moving Average (KAMA) as a filter and 
    classifies regimes using the Mean Square Ratio (MSR).
    
    Math Summary:
    1. Efficiency Ratio (ER) = |Total Price Change| / Sum of Absolute Price Changes
    2. Smoothing Constant (SC) = [ER * (fastestSC - slowestSC) + slowestSC]^2
    3. KAMA_t = KAMA_{t-1} + SC * (Price_t - KAMA_{t-1})
    4. MSR = MeanSquare(diff(KAMA)) / MeanSquare(diff(Price)) over a window
    
    Regime Rules:
    - Bull: MSR > Threshold AND KAMA_t > KAMA_{t-1}
    - Bear: MSR > Threshold AND KAMA_t < KAMA_{t-1}
    - Neutral: MSR <= Threshold
    """
    def __init__(self, config: dict):
        regime_cfg = config.get("regime", {})
        self.er_period = regime_cfg.get("er_period", 10)
        self.msr_period = regime_cfg.get("msr_period", 20)
        self.msr_threshold = regime_cfg.get("msr_threshold", 0.15)
        self.debug = regime_cfg.get("debug", False)
        
        # KAMA constants
        fast_period = 2
        slow_period = 30
        self.fast_sc = 2 / (fast_period + 1)
        self.slow_sc = 2 / (slow_period + 1)

    def _calculate_er(self, prices: np.ndarray, index: int) -> float:
        """Calculates Efficiency Ratio for KAMA smoothing."""
        if index < self.er_period:
            return 0.0
        
        window = prices[index - self.er_period : index + 1]
        change = abs(window[-1] - window[0])
        volatility = np.sum(np.abs(np.diff(window)))
        
        return change / volatility if volatility != 0 else 0.0

    def calculate_kama(self, prices: np.ndarray) -> np.ndarray:
        """Calculates the KAMA series for a given price array."""
        n = len(prices)
        kama = np.zeros(n)
        if n < self.er_period:
            kama[:] = prices[0] if n > 0 else 0
            return kama
            
        kama[self.er_period-1] = prices[self.er_period-1]

        for i in range(self.er_period, n):
            er = self._calculate_er(prices, i)
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
            
        # Backfill initial values
        kama[:self.er_period-1] = kama[self.er_period-1]
        return kama

    def _calculate_msr(self, prices: np.ndarray, kama: np.ndarray) -> float:
        """
        Calculates the Mean Square Ratio (MSR) per Pomorski.
        MSR = MS(diff(KAMA)) / MS(diff(Price))
        """
        if len(prices) < self.msr_period + 1:
            return 0.0
            
        p_diff = np.diff(prices[-self.msr_period-1:])
        k_diff = np.diff(kama[-self.msr_period-1:])
        
        ms_price = np.mean(p_diff ** 2)
        ms_kama = np.mean(k_diff ** 2)
        
        return ms_kama / ms_price if ms_price != 0 else 0.0

    def predict_state(self, prices: np.ndarray) -> str:
        """Predicts market regime (Bull/Bear/Neutral)."""
        if len(prices) < max(self.er_period, self.msr_period) + 5:
            return "Neutral"

        kama = self.calculate_kama(prices)
        msr = self._calculate_msr(prices, kama)
        
        # Check slope direction
        kama_trending_up = kama[-1] > kama[-2]
        
        if self.debug:
            logger.debug(f"KAMA-MSR: MSR={msr:.4f} KAMA_diff={kama[-1]-kama[-2]:.4f}")

        if msr <= self.msr_threshold:
            return "Neutral"
        
        return "Bull" if kama_trending_up else "Bear"

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
        slope, _ = np.polyfit(x, y, 1)
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
