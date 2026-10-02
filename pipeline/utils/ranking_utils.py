from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("../../datasets/findhr")

B_COLS = ["fit_edu", "fit_years", "fit_lang"]
K_PLACEBO = 80
SEED = 42
K_FOLDS = 5

KBS = ("esco", "onet")

# Skill column prefix of each arm family; the column of a KB is f"{prefix}_{kb}".
KB_METHODS = {
    "SY": "fit_skills_sy",
    "KB_hop": "fit_skills_kb_hop",
    "KB_neighbor": "fit_skills_kb_neighbor",
    "Transformer": "fit_skills_transformer",
}

# The placebo is a shuffle of the ESCO hop scores.
PLACEBO_SOURCE = "fit_skills_kb_hop_esco"
GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne)
        for nl in (15, 31) for lr in (0.05, 0.1) for ne in (100, 300)]

# Arm families in the order of the paper's tables.
ARM_ORDER = ["B_only", "Placebo", "SY_raw", "SY", "KB_neighbor", "KB_hop", "Transformer", "Transformer_raw"]
FAMILY_LABELS = {
    "B_only": r"$T_{BN}$", "Placebo": r"$T_{PL}$", "SY_raw": r"$T_{RW}$", "SY": r"$T_{SY}$",
    "KB_neighbor": r"$T_{KB}^{(ne)}$", "KB_hop": r"$T_{KB}^{(hop)}$",
    "Transformer": r"$T_{TF}^{(n)}$", "Transformer_raw": r"$T_{TF}^{(r)}$",
}

# All twelve arms: the KB-free controls, the ESCO block, the O*NET block, the raw transformer.
FULL_ARMS = ARM_ORDER[:3] + [f"{a}_{kb}" for kb in KBS for a in ARM_ORDER if a in KB_METHODS] + ARM_ORDER[-1:]

# Judge folder under datasets/findhr/judges, and its Ollama model.
JUDGES = {
    "qwen3-8b": "qwen3:8b",
    "qwen3-14b": "qwen3:14b",
    "ministral-3-14b": "ministral-3:14b",
}


def kb_arms(kb):
    """The eight arms of one KB, in ARM_ORDER."""
    return [f"{a}_{kb}" if a in KB_METHODS else a for a in ARM_ORDER]


def make_arms():
    """Feature columns of every arm."""
    return {
        "Placebo": B_COLS + ["fit_skills_placebo"],
        "B_only": list(B_COLS),
        "SY_raw": B_COLS + ["fit_skills_raw"],
        "Transformer_raw": B_COLS + ["fit_skills_transformer_raw"],
    } | {f"{name}_{kb}": B_COLS + [f"{col}_{kb}"] for name, col in KB_METHODS.items() for kb in KBS}


def load_frame(judge):
    """Fitness matrix joined with the judge's ground truth, sorted by query.

    `cv_folds` and `normalized_positions` rely on this row order.
    """
    return (pd.read_csv(DATA_DIR / "fitness_matrix.csv")
              .merge(pd.read_csv(DATA_DIR / "judges" / judge / "ground_truth.csv"), on=["id_j", "id_c"])
              .sort_values("id_j")
              .reset_index(drop=True))


def placebo_frame(df, seed):
    """`df` with the placebo column set to permutation `seed` of PLACEBO_SOURCE."""
    return df.assign(fit_skills_placebo=df[PLACEBO_SOURCE].sample(frac=1, random_state=seed).values)


def cv_folds(df, k=5, seed=42):
    """Query-disjoint k-fold: each yield maps "train", "valid" and "test" to (rows, group sizes).

    Every query is tested once. The other k-1 folds are split once into train and valid
    (75/25) for the grid search.
    """
    shuffled = np.random.default_rng(seed).permutation(np.sort(df["id_j"].unique()))
    for test_q in np.array_split(shuffled, k):
        rest = shuffled[~np.isin(shuffled, test_q)]
        train_q, valid_q = train_test_split(rest, test_size=0.25, random_state=seed)
        parts = {}
        for fold, queries in (("train", train_q), ("valid", valid_q), ("test", test_q)):
            sub = df[df["id_j"].isin(queries)].sort_values(by=["id_j", "id_c"])
            parts[fold] = (sub, sub.groupby("id_j").size().values)
        yield parts


def train_select(cols, parts, seed=42):
    """Grid-search an LGBMRanker on the valid nDCG@10 and return the best model."""
    (tr, gtr), (va, gva) = parts["train"], parts["valid"]
    best_ndcg, best_model = -1.0, None
    for params in GRID:
        m = lgb.LGBMRanker(objective="lambdarank", random_state=seed, verbose=-1, n_jobs=-1,
                           force_row_wise=True, **params)
        m.fit(tr[cols], tr["relevance"], group=gtr)
        ndcg = np.mean(per_query_metrics(va, gva, m.predict(va[cols]))["ndcg@10"])
        if ndcg > best_ndcg:
            best_ndcg, best_model = ndcg, m
    return best_model


def run_arm(cols, folds, seed):
    """Train and test one arm on every fold.

    Returns {"cols", "models" (one per fold), "test" (per-query metrics, folds concatenated),
    "test_ids" (the id_j of each test position)}. Arms that share `folds` share the query order.
    """
    models, per_fold, ids = [], [], []
    for parts in folds:
        m = train_select(cols, parts, seed)
        tsub, tg = parts["test"]
        models.append(m)
        per_fold.append(per_query_metrics(tsub, tg, m.predict(tsub[cols])))
        ids.append(tsub["id_j"].unique())
    test = {k: np.concatenate([f[k] for f in per_fold]) for k in per_fold[0]}
    return {"cols": cols, "models": models, "test": test, "test_ids": np.concatenate(ids)}


def per_query_metrics(sub, group, scores):
    """Per-query nDCG@10, and `has_rel`: whether the query has any relevant candidate."""
    bounds = np.cumsum(group)[:-1]
    rels = np.split(sub["relevance"].values, bounds)
    return {"ndcg@10": np.array([ndcg_score([r], [s], k=10) for r, s in zip(rels, np.split(scores, bounds))]),
            "has_rel": np.array([bool((r > 0).any()) for r in rels])}


def mean_metrics(metrics):
    """Mean of each metric over the queries with a relevant candidate."""
    keep = metrics["has_rel"]
    return {m: float(v[keep].mean()) for m, v in metrics.items() if m != "has_rel"}
