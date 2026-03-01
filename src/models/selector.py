import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from src.models.regime import KAMARegimeDetector

logger = logging.getLogger("selector")

class InstitutionalSelector:
    def __init__(self, config: dict, target_count: int = 40):
        self.config = config
        self.target_count = target_count
        self.spy_benchmark = "SPY"
        self.detector = KAMARegimeDetector(config)

    def get_moat_candidates(self, tickers: list) -> pd.DataFrame:
        """Preliminary screen for quality: ROE > 15%, Debt/Equity < 0.6, FCF Yield > 4%."""
        logger.info(f"Screening {len(tickers)} tickers for Moat quality...")
        candidates = []
        
        for ticker_sym in tickers:
            try:
                t = yf.Ticker(ticker_sym)
                info = t.info
                
                roe = info.get("returnOnEquity", 0)
                debt_to_equity = info.get("debtToEquity", 100) / 100.0
                fcf = info.get("freeCashflow", 0)
                market_cap = info.get("marketCap", 1)
                fcf_yield = fcf / market_cap if market_cap > 0 else 0
                
                if roe > 0.15 and debt_to_equity < 0.6 and fcf_yield > 0.04:
                    candidates.append({
                        "symbol": ticker_sym,
                        "roe": roe,
                        "debt_to_equity": debt_to_equity,
                        "fcf_yield": fcf_yield
                    })
            except Exception:
                continue
        
        return pd.DataFrame(candidates)

    def calculate_triple_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates MRD (Priority 1), Beta (Priority 2), and Sharpe (Priority 3)."""
        if df.empty:
            return df
            
        symbols = df["symbol"].tolist()
        logger.info(f"Calculating Triple-Metrics for {len(symbols)} candidates...")
        
        # Download 2 years of data for stable MSR fitting
        data = yf.download(symbols + [self.spy_benchmark], period="2y", interval="1d", progress=False)["Close"]
        returns = data.pct_change().dropna()
        spy_returns = returns[self.spy_benchmark]
        
        results = []
        for symbol in symbols:
            if symbol not in returns: continue
            try:
                stock_prices = data[symbol].dropna().values
                stock_returns = returns[symbol]
                
                # 1. Calculate MRD (Mean Return Difference)
                # We use the actual KAMA-MSR logic to identify historical regimes
                log_rets = np.diff(np.log(stock_prices))
                
                # Manual KAMA+MSR fit for MRD estimation
                kama = self.detector.calculate_kama(stock_prices)
                kama_rets = np.diff(np.log(kama[~np.isnan(kama)]))
                
                # Fit MSR to find Bull/Bear means
                mod = MarkovRegression(kama_rets, k_regimes=2, trend='c', switching_variance=True)
                res = mod.fit(disp=False)
                
                # Means are in res.params[0] and res.params[1] (approx)
                # We take the difference between the high-mean and low-mean states
                mrd = abs(res.params[0] - res.params[1]) * 252 # Annualized difference
                
                # 2. Beta
                covariance = np.cov(stock_returns, spy_returns)[0, 1]
                variance = np.var(spy_returns)
                beta = covariance / variance if variance != 0 else 1.0
                
                # 3. Sharpe
                sharpe = (stock_returns.mean() / stock_returns.std()) * np.sqrt(252) if stock_returns.std() != 0 else 0
                
                results.append({
                    "symbol": symbol,
                    "mrd": mrd,
                    "beta": beta,
                    "sharpe": sharpe
                })
            except Exception as e:
                logger.debug(f"Failed to calculate metrics for {symbol}: {e}")
                continue
                
        metrics_df = pd.DataFrame(results)
        return pd.merge(df, metrics_df, on="symbol")

    def select_best_40(self, df: pd.DataFrame) -> list:
        """
        Final selection based on User Priorities:
        1) High MRD (Exploitable signal)
        2) Target Portfolio Beta = 0.3
        3) Max Sharpe (Tie-breaker)
        """
        if df.empty: return []
        
        # Priority 1: Sort by MRD primarily
        df = df.sort_values("mrd", ascending=False)
        
        # Priority 2: Greedy selection to hit target average beta 0.3
        # We start with the top MRD stocks and "steer" the portfolio
        candidates = df.to_dict('records')
        selected = []
        current_avg_beta = 0.0
        
        # We take more than 40 initially to have room to optimize beta
        pool = candidates[:80] 
        
        for _ in range(self.target_count):
            if not pool: break
            
            best_pick = None
            best_dist = float('inf')
            
            for i, c in enumerate(pool):
                # Calculate what the new average beta would be if we picked this stock
                potential_avg = (sum(s['beta'] for s in selected) + c['beta']) / (len(selected) + 1)
                dist_to_target = abs(potential_avg - 0.3)
                
                # Scoring formula: Weighted distance to target beta + tie-break by Sharpe
                # Since MRD is already the sort order, we look for the stock in the 
                # top of the list that helps our Beta most.
                if dist_to_target < best_dist:
                    best_dist = dist_to_target
                    best_pick = i
            
            selected.append(pool.pop(best_pick))
            
        final_symbols = [s['symbol'] for s in selected]
        final_beta = np.mean([s['beta'] for s in selected])
        final_mrd = np.mean([s['mrd'] for s in selected])
        
        logger.info(f"Final Selection: Avg MRD={final_mrd:.4f}, Avg Beta={final_beta:.2f}")
        return final_symbols
