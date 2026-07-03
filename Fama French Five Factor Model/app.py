import os
import json
import logging
import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# --- Institutional Operational Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# --- Configuration ---
TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = '2023-01-01'
RISK_FREE_RATE = 0.04
TRANS_COST = 0.0010
TURNOVER_PENALTY_LAMBDA = 0.0050
PORTFOLIO_AUM = 100_000_000
MAX_ADV_PARTICIPATION = 0.01
STATE_FILE = "portfolio_state.json"

def fetch_and_clean_production_data(tickers, start_date):
    """Ingests and synchronizes Fama-French factors and asset pricing."""
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date)[0]
    ff_data.index = ff_data.index.to_timestamp()
    ff_data = ff_data / 100
    
    raw_data = yf.download(tickers, start=start_date, auto_adjust=True, group_by='ticker', progress=False)
    
    asset_returns = pd.DataFrame()
    asset_adv = pd.DataFrame()
    
    for ticker in tickers:
        # Resolve MultiIndex: if single ticker, yf returns simple DataFrame, else MultiIndex
        df_ticker = raw_data[ticker] if len(tickers) > 1 else raw_data
        asset_returns[ticker] = df_ticker['Close'].pct_change()
        asset_adv[ticker] = (df_ticker['Close'] * df_ticker['Volume']).rolling(20).mean()
            
    master_df = ff_data.join(asset_returns, how='inner').ffill().dropna()
    master_adv = asset_adv.reindex(master_df.index).ffill()
    
    return master_df, master_adv, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

def optimize_production_portfolio(tickers, mean_returns, cov_matrix, risk_free, betas, current_weights, adv_limits):
    num_assets = len(tickers)
    init_weights = np.ones(num_assets) / num_assets if current_weights is None else current_weights
    
    bounds = tuple([(-0.5, 1.5) for _ in range(num_assets)]) # Simplified constraints

    def analytical_objective(weights):
        p_ret = np.sum(mean_returns * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        prev_w = np.zeros(num_assets) if current_weights is None else current_weights
        turnover_drag = TURNOVER_PENALTY_LAMBDA * np.sum(np.square(weights - prev_w))
        return -((p_ret - risk_free) / p_vol - turnover_drag) if p_vol > 1e-6 else 0

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.sum(w * betas) - 0.0}
    ]
    
    res = minimize(analytical_objective, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else init_weights

# --- Execution ---
if _name_ == "_main_":
    data_df, adv_df, factors = fetch_and_clean_production_data(TICKERS, START_DATE)
    
    # Snapshot last 126 days (6 months)
    is_df = data_df.iloc[-126:]
    lw_cov = LedoitWolf().fit(is_df[TICKERS]).covariance_
    mean_rets = is_df[TICKERS].mean().values
    
    betas = []
    for t in TICKERS:
        model = sm.OLS(is_df[t] - is_df['RF'], sm.add_constant(is_df[factors])).fit()
        betas.append(model.params['Mkt-RF'])
        
    weights = optimize_production_portfolio(TICKERS, mean_rets, lw_cov, RISK_FREE_RATE, np.array(betas), None, None)
    
    print(f"Optimization complete. Optimized Weights: {dict(zip(TICKERS, weights))}")