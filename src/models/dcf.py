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
    """Calculates position size using Rolling Kelly and Volatility Scaling."""
    def __init__(self, config: dict):
        self.kelly_fraction = config["sizing"]["kelly_fraction"]
        self.vix_baseline = config["sizing"]["vix_baseline"]
        self.vix_max = config["sizing"]["vix_max"]
        self.atr_baseline = config["sizing"]["atr_baseline_pct"]
        self.atr_max = config["sizing"]["atr_max_pct"]
        self.adv_limit = config["risk"]["adv_position_limit"]
        
        # New Hardening Parameters
        self.min_samples = config["sizing"].get("min_kelly_samples", 10)
        self.max_cap = config["sizing"].get("max_kelly_cap", 0.20)
        self.default_win_rate = config["sizing"].get("default_win_rate", 0.50)
        self.default_win_loss = config["sizing"].get("default_win_loss", 1.2)

    def calculate_kelly(self, realized_trades: list = None) -> float:
        """
        Calculates Kelly Criterion based on realized performance.
        f* = (bp - q) / b
        """
        if not realized_trades or len(realized_trades) < self.min_samples:
            p = self.default_win_rate
            b = self.default_win_loss
        else:
            # Calculate from realized outcomes
            # realized_trades is a list of returns (e.g. 0.05, -0.02)
            wins = [r for r in realized_trades if r > 0]
            losses = [r for r in realized_trades if r <= 0]
            
            p = len(wins) / len(realized_trades)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1e-6
            b = avg_win / avg_loss if avg_loss > 0 else self.default_win_loss

        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0
        
        # Apply fractional multiplier and strict cap
        final_f = min(kelly_f * self.kelly_fraction, self.max_cap)
        return max(0.0, final_f)

    def apply_vol_scaling(self, base_f: float, current_vix: float, current_atr_pct: float) -> float:
        """Reduces position size as volatility (VIX/ATR) rises."""
        # VIX Scale: 1.0 at baseline, 0.0 at max
        vix_denom = self.vix_max - self.vix_baseline
        vix_scale = max(0.0, 1.0 - (current_vix - self.vix_baseline) / vix_denom) if vix_denom > 0 else 1.0
        
        # ATR Scale: 1.0 at baseline, 0.0 at max
        atr_denom = self.atr_max - self.atr_baseline
        atr_scale = max(0.0, 1.0 - (current_atr_pct - self.atr_baseline) / atr_denom) if atr_denom > 0 else 1.0
        
        combined_scale = min(vix_scale, atr_scale)
        return base_f * combined_scale

    def final_shares(self, equity: float, price: float, f: float, adv_20: float) -> int:
        """Calculates final share count, capped by 20-day Average Daily Volume."""
        dollar_amount = equity * f
        shares = int(dollar_amount / price)
        
        # ADV Cap: Don't be more than 1% of daily volume (or as configured)
        max_shares = int(adv_20 * self.adv_limit)
        
        return min(shares, max_shares)
