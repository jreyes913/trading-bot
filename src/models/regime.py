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
    Implements the optimal KAMA-MSR Regime-Switching Framework.
    Replication of Piotr Pomorski, UCL PhD Thesis (2024).
    
    Logic:
    1. 2-state MSR on log-returns -> Detects Low/High Variance states.
    2. KAMA filter overlay -> Detects Bullish/Bearish direction.
    3. Combinatorial mapping to 3 optimal states (Bullish, Bearish, Other).
    
    Note: Following Pomorski's 'Optimal' discovery, this detector uses 
    CONTRARIAN mapping for the bot's consumption:
    - Market 'Low-Var Bullish' (Label 1) -> Returns 'Bear' (Trigger Short/Exit)
    - Market 'High-Var Bearish' (Label 2) -> Returns 'Bull' (Trigger Long)
    - Others (Label 0) -> Returns 'Neutral' (Flat)
    """
    def __init__(self, config: dict):
        regime_cfg = config.get("regime", {})
        self.n = regime_cfg.get("er_period", 10)
        self.msr_window = regime_cfg.get("msr_window", 100)
        self.gamma = regime_cfg.get("gamma", 0.15)
        self.tc = regime_cfg.get("tc", 0.0040)
        self.debug = regime_cfg.get("debug", False)
        
        # KAMA constants
        self.n_s = 2
        self.n_l = 30
        self.fast_sc = 2.0 / (self.n_s + 1)
        self.slow_sc = 2.0 / (self.n_l + 1)

    def calculate_kama(self, prices: np.ndarray) -> np.ndarray:
        """Pure-Python KAMA implementation from sample.py."""
        n = len(prices)
        kama = np.full(n, np.nan)
        if n < self.n:
            return kama
            
        kama[self.n - 1] = prices[self.n - 1]
        for i in range(self.n, n):
            direction = abs(prices[i] - prices[i - self.n])
            volatility = np.sum(np.abs(np.diff(prices[i - self.n : i + 1])))
            er = direction / volatility if volatility != 0 else 0.0
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])
        return kama

    def predict_state(self, prices: np.ndarray) -> str:
        """
        Detects the current regime using MSR variance and KAMA direction.
        """
        if len(prices) < self.msr_window + self.n:
            return "Neutral"

        try:
            # 0. Quick Volatility Check
            if np.std(prices[-self.n:]) < 1e-6:
                return "Neutral"

            # 1. MSR for Variance State
            log_ret = np.diff(np.log(prices))[-self.msr_window:]
            
            model = MarkovRegression(log_ret, k_regimes=2, trend='c', switching_variance=True)
            res = model.fit(disp=False)
            
            # Identify low-variance regime by smallest conditional sigma2
            v0 = res.params[-2]
            v1 = res.params[-1]
            low_var_idx = 0 if v0 < v1 else 1
            
            probs = res.smoothed_marginal_probabilities
            p_low = probs[-1, low_var_idx] if hasattr(probs, "shape") else probs.iloc[-1].values[low_var_idx]

            # 2. KAMA for Directional Signal
            kama = self.calculate_kama(prices)
            kama_diffs = np.diff(kama[-self.n*2:])
            # Standard deviation of changes
            std_diff = np.std(kama_diffs[-self.n:])
            filt = self.gamma * std_diff
            
            latest_kama = kama[-1]
            rolling_min = np.min(kama[-self.n:])
            rolling_max = np.max(kama[-self.n:])
            
            kama_sig = 0
            # Ensure meaningful movement relative to price scale
            if std_diff > 1e-7:
                if latest_kama < rolling_max - filt:
                    kama_sig = -1
                elif latest_kama > rolling_min + filt:
                    kama_sig = 1

            # 3. Combinatorial Mapping (Chapter 4)
            is_lv_bull = (p_low > 0.5) and (kama_sig > 0)
            is_hv_bear = (p_low <= 0.5) and (kama_sig < 0)

            if self.debug:
                logger.info(f"event=REGIME_DEBUG p_low={p_low:.2f} kama_sig={kama_sig} latest={latest_kama:.2f} min={rolling_min:.2f} max={rolling_max:.2f} filt={filt:.4f}")

            # 4. Contrarian Output for Bot Consumption
            if is_hv_bear:
                return "Bull"
            elif is_lv_bull:
                return "Bear"
            else:
                return "Neutral"

        except Exception as e:
            logger.error(f"event=MSR_OPTIMAL_FAILURE error='{e}'")
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
