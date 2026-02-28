import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("selector")

class InstitutionalSelector:
    def __init__(self, target_count: int = 40):
        self.target_count = target_count
        self.spy_benchmark = "SPY"

    def get_moat_candidates(self, tickers: list) -> pd.DataFrame:
        """Filters for institutional 'Moat' quality using yfinance."""
        logger.info(f"Checking moat metrics for {len(tickers)} tickers...")
        candidates = []
        
        # We process in batches to avoid rate limits
        for ticker_sym in tickers:
            try:
                t = yf.Ticker(ticker_sym)
                info = t.info
                
                roe = info.get("returnOnEquity", 0)
                debt_to_equity = info.get("debtToEquity", 100) / 100.0 # yf returns 50 for 0.5
                fcf = info.get("freeCashflow", 0)
                market_cap = info.get("marketCap", 1)
                fcf_yield = fcf / market_cap if market_cap > 0 else 0
                
                # Institutional Moat Filter: ROE > 15%, Debt/Equity < 0.6, FCF Yield > 4%
                if roe > 0.15 and debt_to_equity < 0.6 and fcf_yield > 0.04:
                    candidates.append({
                        "symbol": ticker_sym,
                        "roe": roe,
                        "debt_to_equity": debt_to_equity,
                        "fcf_yield": fcf_yield
                    })
                    logger.info(f"Moat candidate found: {ticker_sym}")
            except Exception:
                continue
                
            if len(candidates) >= 100: # Limit initial pool for speed
                break
                
        return pd.DataFrame(candidates)

    def calculate_quant_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Beta (vs SPY) and Sharpe Ratio."""
        if df.empty:
            return df
            
        tickers = df["symbol"].tolist()
        # Download 1 year of data
        data = yf.download(tickers + [self.spy_benchmark], period="1y", interval="1d", progress=False)["Close"]
        returns = data.pct_change().dropna()
        
        spy_returns = returns[self.spy_benchmark]
        
        results = []
        for ticker in tickers:
            if ticker not in returns: continue
            stock_returns = returns[ticker]
            
            # Beta calculation
            covariance = np.cov(stock_returns, spy_returns)[0, 1]
            variance = np.var(spy_returns)
            beta = covariance / variance if variance != 0 else 1.0
            
            # Sharpe Ratio
            sharpe = (stock_returns.mean() / stock_returns.std()) * np.sqrt(252) if stock_returns.std() != 0 else 0
            
            results.append({
                "symbol": ticker,
                "beta": beta,
                "sharpe": sharpe
            })
            
        quant_df = pd.DataFrame(results)
        return pd.merge(df, quant_df, on="symbol")

    def select_best_40(self, df: pd.DataFrame) -> list:
        """Selects 40 stocks targeting average Beta of 0.3 while maximizing Sharpe."""
        if df.empty: return []
        if len(df) <= self.target_count:
            return df["symbol"].tolist()
            
        # Strategy: Primary sort by Sharpe, secondary by Beta proximity to 0.3
        df["beta_diff"] = abs(df["beta"] - 0.3)
        final_selection = df.sort_values(["beta_diff", "sharpe"], ascending=[True, False]).head(self.target_count)
        
        logger.info(f"Final Selection Avg Beta: {final_selection['beta'].mean():.2f}")
        return final_selection["symbol"].tolist()
