import numpy as np
import logging

logger = logging.getLogger("dcf")

class DCFValuator:
    """Performs Monte Carlo DCF simulations to find intrinsic value."""
    def __init__(self, config: dict):
        self.simulations = config["validation"]["dcf_simulations"]
        self.undervalue_threshold = config["validation"]["dcf_undervalue_threshold"]

    def estimate_value(self, ebitda: float, growth_mean: float, growth_std: float) -> float:
        """
        Simulates intrinsic value based on EBITDA and projected growth.
        This is a simplified version of the Monte Carlo DCF.
        """
        # Run 10,000 simulations of growth paths
        sim_growth = np.random.normal(growth_mean, growth_std, self.simulations)
        
        # Simple multiple-based valuation for demonstration
        # In a real build, this would project cash flows for 5-10 years and discount them
        multiples = np.random.normal(12.0, 2.0, self.simulations)
        valuations = ebitda * (1 + sim_growth) * multiples
        
        return float(np.median(valuations))

    def is_undervalued(self, current_price: float, estimated_value: float) -> bool:
        """Checks if the stock is trading at a significant discount."""
        return current_price < (estimated_value * self.undervalue_threshold)

class PositionSizer:
    """Calculates position size using Fractional Kelly and Volatility Scaling."""
    def __init__(self, config: dict):
        self.kelly_fraction = config["sizing"]["kelly_fraction"]
        self.vix_baseline = config["sizing"]["vix_baseline"]
        self.vix_max = config["sizing"]["vix_max"]
        self.atr_baseline = config["sizing"]["atr_baseline_pct"]
        self.atr_max = config["sizing"]["atr_max_pct"]
        self.adv_limit = config["risk"]["adv_position_limit"]

    def calculate_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """Standard Kelly Criterion: f* = (bp - q) / b"""
        b = win_loss_ratio
        p = win_rate
        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0
        return max(0, kelly_f * self.kelly_fraction)

    def apply_vol_scaling(self, base_f: float, current_vix: float, current_atr_pct: float) -> float:
        """Reduces position size as volatility (VIX/ATR) rises."""
        # VIX Scale: 1.0 at baseline, 0.0 at max
        vix_scale = max(0.0, 1.0 - (current_vix - self.vix_baseline) / (self.vix_max - self.vix_baseline))
        
        # ATR Scale: 1.0 at baseline, 0.0 at max
        atr_scale = max(0.0, 1.0 - (current_atr_pct - self.atr_baseline) / (self.atr_max - self.atr_baseline))
        
        combined_scale = min(vix_scale, atr_scale)
        return base_f * combined_scale

    def final_shares(self, equity: float, price: float, f: float, adv_20: float) -> int:
        """Calculates final share count, capped by 20-day Average Daily Volume."""
        dollar_amount = equity * f
        shares = int(dollar_amount / price)
        
        # ADV Cap: Don't be more than 1% of daily volume
        max_shares = int(adv_20 * self.adv_limit)
        
        return min(shares, max_shares)
