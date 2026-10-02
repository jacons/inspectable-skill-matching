import numpy as np
import pandas as pd
from scipy.stats import permutation_test, rankdata


def normalized_positions(scores, group):
    """Position of each candidate in its query, from 0 (top) to 1 (bottom); NaN for a lone candidate.

    Ties get the average rank, so every query keeps a mean position of 0.5.
    """
    out = []
    for s in np.split(np.asarray(scores), np.cumsum(group)[:-1]):
        t = rankdata(-s, method="average")
        out.append((t - 1) / (len(s) - 1) if len(s) > 1 else np.full(len(s), np.nan))
    return np.concatenate(out)


def per_query_meanpos(dt, p, grade=2):
    """Mean of `p` (aligned to the rows of `dt`) over the grade-`grade` candidates of each query.

    Only queries with at least one candidate of that grade appear in the index.
    """
    m = (dt["relevance"] == grade).values
    return pd.Series(p[m], index=dt.loc[m, "id_j"].values).groupby(level=0).mean()


def perm_pvalue(differences):
    """Two-sided paired sign-flip permutation test on per-query differences (20,000 resamples)."""
    return permutation_test(
        (differences, np.zeros_like(differences)),
        statistic=lambda d, z, axis: np.mean(d - z, axis=axis),
        permutation_type="samples", alternative="two-sided",
        vectorized=True, n_resamples=20_000, random_state=42,
    ).pvalue


def pair_pvalue(xa, xb):
    """p-value of arm a vs arm b from their per-query nDCG, each (n,) or (n, K) for the placebo.

    With K replicates the test runs on each one and returns twice the median p-value (Rueger).
    """
    col = lambda x: np.atleast_2d(np.asarray(x, dtype=float).T).T
    ps = [perm_pvalue(d) for d in (col(xa) - col(xb)).T]
    return ps[0] if len(ps) == 1 else min(1.0, 2 * float(np.median(ps)))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    out = np.empty(m)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def tie_frac(v):
    """Share of candidate pairs of one query with identical feature vectors; NaN for a lone candidate."""
    c = v.value_counts().values
    n = len(v)
    return np.nan if n < 2 else (c * (c - 1)).sum() / (n * (n - 1))
