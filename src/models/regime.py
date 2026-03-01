import numpy as np
import pandas as pd
import logging
import warnings
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Suppress statsmodels convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

logger = logging.getLogger("regime")

class KAMARegimeDetector:
    """
    Implements KAMA-MSR (Kaufman Adaptive Moving Average + Markov-Switching Regression).
    Based on Piotr Pomorski's research.
    """
    def __init__(self, config: dict):
        regime_cfg = config.get("regime", {})
        self.er_period = regime_cfg.get("er_period", 10)
        self.msr_window = regime_cfg.get("msr_window", 100)
        self.msr_regimes = regime_cfg.get("msr_regimes", 3)
        self.debug = regime_cfg.get("debug", False)
        
        self.fast_sc = 2 / (2 + 1)
        self.slow_sc = 2 / (30 + 1)

    def _calculate_er(self, prices: np.ndarray, index: int) -> float:
        if index < self.er_period: return 0.0
        window = prices[index - self.er_period : index + 1]
        change = abs(window[-1] - window[0])
        volatility = np.sum(np.abs(np.diff(window)))
        return change / volatility if volatility != 0 else 0.0

    def calculate_kama(self, prices: np.ndarray) -> np.ndarray:
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
        kama[:self.er_period-1] = kama[self.er_period-1]
        return kama

    def predict_state(self, prices: np.ndarray) -> str:
        """
        Uses MSR to identify latent states and KAMA direction for confirmation.
        """
        if len(prices) < self.er_period + 10:
            return "Neutral"

        kama = self.calculate_kama(prices)
        kama_slope = (kama[-1] - kama[-5]) / kama[-5] if kama[-5] != 0 else 0
        latest_er = self._calculate_er(prices, len(prices) - 1)

        # 1. Structural Filter: Higher threshold for 'Trending' vs 'Sideways'
        if latest_er < 0.3: return "Neutral"

        # 2. Econometric Classification (MSR)
        if len(prices) >= self.msr_window + self.er_period:
            try:
                kama_series = pd.Series(kama)
                kama_returns = np.log(kama_series / kama_series.shift(1)).dropna().values[-self.msr_window:]
                
                # Defensive check for data quality
                if len(kama_returns) < self.msr_window or not np.all(np.isfinite(kama_returns)):
                    raise ValueError(f"kama_returns invalid or insufficient (len={len(kama_returns)})")

                model = MarkovRegression(kama_returns, k_regimes=self.msr_regimes, switching_variance=False)
                res = model.fit(disp=False)
                
                # Extract means via parameter names to ensure reliability
                means = [res.params[name] for name in res.param_names if 'const' in name]
                if len(means) != self.msr_regimes:
                    raise ValueError(f"Extracted {len(means)} means, expected {self.msr_regimes}")

                bull_idx = np.argmax(means)
                bear_idx = np.argmin(means)
                
                latest_probs = res.smoothed_marginal_probabilities[-1]
                bull_prob = latest_probs[bull_idx]
                bear_prob = latest_probs[bear_idx]

                # Double-Lock confirmation
                if bull_prob > 0.6 and means[bull_idx] > 0 and kama_slope > 0.0001:
                    return "Bull"
                if bear_prob > 0.6 and means[bear_idx] < 0 and kama_slope < -0.0001:
                    return "Bear"
                
            except Exception as e:
                logger.error(
                    f"event=MSR_FIT_FAILURE error='{e}' "
                    f"msr_window={self.msr_window} msr_regimes={self.msr_regimes} latest_er={latest_er:.4f}"
                )

        # 3. Aggressive Fallback (if trending strongly but MSR failed or inconclusive)
        if kama_slope > 0.001: return "Bull"
        if kama_slope < -0.001: return "Bear"
        
        return "Neutral"

class FastExitOverlay:
    def __init__(self, config: dict):
        self.lookback = config["regime"]["fast_exit_lookback_bars"]
        self.threshold = config["regime"]["fast_exit_slope_threshold"]

    def calculate_slope(self, spy_closes: pd.Series) -> float:
        if len(spy_closes) < self.lookback: return 0.0
        y = spy_closes.iloc[-self.lookback:].values
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope / y.mean()

    def should_fast_exit(self, current_regime: str, spy_closes: pd.Series) -> bool:
        if current_regime != "Bull": return False
        slope = self.calculate_slope(spy_closes)
        return slope < self.threshold
