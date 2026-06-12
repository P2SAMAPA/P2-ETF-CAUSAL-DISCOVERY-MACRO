import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations

def partial_correlation(data, i, j, cond_set):
    """Compute partial correlation between variables i and j given conditioning set."""
    cols = [i, j] + list(cond_set)
    if len(cols) > len(data.columns):
        return 1.0, 1.0
    sub = data.iloc[:, cols].dropna()
    if len(sub) < len(cols) + 2:
        return 1.0, 1.0
    # Compute residuals using linear regression
    from sklearn.linear_model import LinearRegression
    # Residual of i on cond_set
    if len(cond_set) > 0:
        X_cond = sub.iloc[:, 2:].values
        reg_i = LinearRegression().fit(X_cond, sub.iloc[:, 0].values)
        resid_i = sub.iloc[:, 0].values - reg_i.predict(X_cond)
        reg_j = LinearRegression().fit(X_cond, sub.iloc[:, 1].values)
        resid_j = sub.iloc[:, 1].values - reg_j.predict(X_cond)
    else:
        resid_i = sub.iloc[:, 0].values
        resid_j = sub.iloc[:, 1].values
    # Correlation of residuals
    corr, p_value = pearsonr(resid_i, resid_j)
    return abs(corr), p_value

def is_independent(data, i, j, cond_set, alpha=0.05):
    """Test conditional independence using partial correlation."""
    _, p = partial_correlation(data, i, j, cond_set)
    return p > alpha

def pc_algorithm(data, alpha=0.05, max_cond_size=3):
    """
    PC algorithm (simplified) to learn a skeleton (undirected graph).
    Returns adjacency matrix (n x n) where 1 indicates an edge.
    """
    n = data.shape[1]
    # Start with complete graph
    adj = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    # Phase I: adjacency search
    for depth in range(max_cond_size + 1):
        changed = False
        for i in range(n):
            neighbors = [j for j in range(n) if adj[i, j] == 1 and j != i]
            for j in neighbors:
                if j <= i:
                    continue
                # Find all subsets of neighbors of size depth (excluding j)
                other_neighbors = [k for k in neighbors if k != j]
                if len(other_neighbors) < depth:
                    continue
                for cond_set in combinations(other_neighbors, depth):
                    if is_independent(data, i, j, cond_set, alpha):
                        adj[i, j] = adj[j, i] = 0
                        changed = True
                        break
                if adj[i, j] == 0:
                    break
        if not changed:
            break
    return adj

def orient_edges(data, adj):
    """
    Orient edges based on v-structures (simplified orientation).
    Returns directed adjacency matrix (1 = i->j).
    """
    n = adj.shape[0]
    directed = adj.copy()
    # Find v-structures: i - j - k with no edge i-k, and i and k not independent given j
    for j in range(n):
        neighbors = [i for i in range(n) if adj[i, j] == 1]
        for i, k in combinations(neighbors, 2):
            if adj[i, k] == 0:
                # Test if i and k are conditionally independent given j
                if not is_independent(data, i, k, [j], alpha=0.05):
                    # Orient i -> j <- k (if not already oriented)
                    if directed[i, j] == 1 and directed[j, i] == 1:
                        directed[i, j] = 0
                    if directed[k, j] == 1 and directed[j, k] == 1:
                        directed[k, j] = 0
    return directed

def causal_score(data, macro_indices, etf_indices, alpha=0.05, max_cond_size=3):
    """
    Compute per-ETF score = number of incoming causal edges from macro variables.
    """
    n = data.shape[1]
    # Learn skeleton
    adj = pc_algorithm(data, alpha, max_cond_size)
    # Orient edges (simplified)
    directed = orient_edges(data, adj)
    # Count incoming edges from macro nodes to each ETF
    scores = {etf_idx: 0 for etf_idx in etf_indices}
    for m in macro_indices:
        for e in etf_indices:
            if directed[m, e] == 1:  # macro -> ETF
                scores[e] += 1
    return scores
