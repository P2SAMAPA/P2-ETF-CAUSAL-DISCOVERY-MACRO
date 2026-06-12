import numpy as np
import pandas as pd  # added missing import
from sklearn.feature_selection import mutual_info_regression

def causal_score(data, macro_indices, etf_indices, mi_threshold=0.01):
    """
    Simplified causal score: for each ETF, count how many macro variables have
    mutual information (non‑linear dependence) above a threshold.
    """
    n_etfs = len(etf_indices)
    scores = {etf_idx: 0 for etf_idx in etf_indices}
    for m_idx in macro_indices:
        macro_vals = data.iloc[:, m_idx].values.reshape(-1, 1)
        for e_idx in etf_indices:
            etf_vals = data.iloc[:, e_idx].values
            mask = ~(np.isnan(macro_vals).flatten() | np.isnan(etf_vals))
            if np.sum(mask) < 10:
                continue
            mi = mutual_info_regression(macro_vals[mask], etf_vals[mask], random_state=42)[0]
            if mi > mi_threshold:
                scores[e_idx] += 1
    return scores

def causal_score_aggregate(returns, macro_df, mi_threshold=0.01):
    """
    Compute per-ETF score = number of macro variables with MI > threshold.
    """
    if macro_df is None or macro_df.empty:
        return {ticker: 0.0 for ticker in returns.columns}
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return {ticker: 0.0 for ticker in returns.columns}
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    combined = pd.concat([ret_aligned, macro_aligned], axis=1)
    etf_indices = list(range(len(returns.columns)))
    macro_indices = list(range(len(returns.columns), len(combined.columns)))
    scores = causal_score(combined, macro_indices, etf_indices, mi_threshold)
    tickers = returns.columns
    return {tickers[e]: float(scores[e]) for e in etf_indices}
