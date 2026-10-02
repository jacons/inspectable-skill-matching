"""Aggregate the ranking results of every judge and KB.

    python pipeline/4_evaluation/evaluation_all_judges.py

Reads each judge's results/{ndcg_per_query,positions}.csv and writes
datasets/findhr/final/{arm_summary,pairwise_pvalues,src_summary}.csv for 8_1_final_results.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from pipeline.utils.eval_utils import holm, pair_pvalue, per_query_meanpos, tie_frac  # noqa: E402
from pipeline.utils.ranking_utils import (  # noqa: E402
    DATA_DIR, FULL_ARMS, JUDGES, KBS, K_PLACEBO, load_frame, make_arms, placebo_frame,
)

FINAL_DIR = DATA_DIR / "final"
ARMS = make_arms()
BASELINE = "B_only"
GRADES = (0, 1, 2)

# Values of the paper's arm summary table (qwen3-8b, ESCO): the run must reproduce them.
PAPER_JUDGE = "qwen3-8b"
PAPER = {
    "gamma": {"SY_raw": .157, "SY_esco": .263, "KB_neighbor_esco": .302, "KB_hop_esco": .318,
              "Transformer_esco": .352, "Transformer_raw": .398},
    "ndcg_mean": {"B_only": .275, "Placebo": .248},
    "spr_2": {"B_only": .416, "Transformer_raw": .151},
}
PAPER_TIES = {"B_only": 37.2, "SY_esco": 28.8}


def family_kb(arm):
    """'KB_hop_onet' -> ('KB_hop', 'onet'); an arm with no KB -> (arm, '--')."""
    fam, _, kb = arm.rpartition("_")
    return (fam, kb) if kb in KBS else (arm, "--")


def tie_rates():
    """% of tied candidate pairs per arm. The features do not depend on the judge."""
    df = load_frame(next(iter(JUDGES)))
    rate = lambda f, cols: f.groupby("id_j")[cols].apply(tie_frac).mean()
    placebo = [placebo_frame(df, s) for s in range(K_PLACEBO)]
    return {a: 100 * (np.mean([rate(f, ARMS[a]) for f in placebo]) if a == "Placebo" else rate(df, ARMS[a]))
            for a in FULL_ARMS}


def aggregate(judge, ties):
    """The arm summary, pairwise p-values and SRC rows of one judge."""
    rdir = DATA_DIR / "judges" / judge / "results"
    ndcg = pd.read_csv(rdir / "ndcg_per_query.csv")
    pos = pd.read_csv(rdir / "positions.csv")

    # Keep the fold order of the B_only rows: the permutation test flips signs by position.
    base_rows = ndcg[ndcg.arm == BASELINE]
    ids, has_rel = base_rows.id_j.values, base_rows.has_rel.values
    real = ndcg[ndcg.replicate == -1].pivot(index="id_j", columns="arm", values="ndcg@10").loc[ids]
    placebo = ndcg[ndcg.arm == "Placebo"].pivot(index="id_j", columns="replicate", values="ndcg@10").loc[ids]
    # X feeds the tests (the placebo as its K replicates), x has one number per query.
    X = {a: real[a].values[has_rel] for a in FULL_ARMS if a != "Placebo"} | {"Placebo": placebo.values[has_rel]}
    x = {a: v.mean(axis=1) if v.ndim == 2 else v for a, v in X.items()}

    pairs = [(a, b) for i, a in enumerate(FULL_ARMS) for b in FULL_ARMS[:i]]
    p_raw = np.array([pair_pvalue(X[a], X[b]) for a, b in pairs])
    p_holm = holm(p_raw)
    pairwise = pd.DataFrame({"judge": judge, "arm_a": [a for a, _ in pairs], "arm_b": [b for _, b in pairs],
                             "diff_mean": [np.mean(x[a] - x[b]) for a, b in pairs],
                             "p_raw": p_raw, "p_holm": p_holm})
    p_vs_bn = {a: p for (a, b), p in zip(pairs, p_holm) if b == BASELINE}

    placebo_p = pos[[f"p_placebo_{s:02d}" for s in range(K_PLACEBO)]].values.mean(axis=1)
    P = {a: placebo_p if a == "Placebo" else pos[f"p_{a}"].values for a in FULL_ARMS}
    arm_rows, src_rows = [], []
    for a in FULL_ARMS:
        fam, kb = family_kb(a)
        ci = ttest_1samp(x[a], 0).confidence_interval()
        row = {"judge": judge, "arm": a, "family": fam, "kb": kb, "n_queries": len(x[a]),
               "ndcg_mean": x[a].mean(), "ndcg_sd": x[a].std(ddof=1),
               "ndcg_ci_lo": ci.low, "ndcg_ci_hi": ci.high,
               "gamma": np.nan, "gamma_ci_lo": np.nan, "gamma_ci_hi": np.nan,
               "p_vs_bn": p_vs_bn.get(a, np.nan), "ties": ties[a]} | {
            f"spr_{g}": per_query_meanpos(pos, P[a], g).mean() for g in GRADES}
        if a != BASELINE:
            d = x[a] - x[BASELINE]
            gci = ttest_1samp(d, 0).confidence_interval()
            row |= {"gamma": d.mean(), "gamma_ci_lo": gci.low, "gamma_ci_hi": gci.high}
            for g in GRADES:
                s = per_query_meanpos(pos, P[BASELINE] - P[a], g).values
                sci = ttest_1samp(s, 0).confidence_interval()
                src_rows.append({"judge": judge, "arm": a, "family": fam, "kb": kb, "grade": g,
                                 "src_mean": s.mean(), "ci_lo": sci.low, "ci_hi": sci.high,
                                 "n_queries": len(s)})
        arm_rows.append(row)
    return pd.DataFrame(arm_rows), pairwise, pd.DataFrame(src_rows)


def check_paper(summary):
    """Exit unless the PAPER_JUDGE rows match the values printed in the paper."""
    q = summary[summary.judge == PAPER_JUDGE].set_index("arm")
    bad = [(c, a, q.loc[a, c], v) for c, vals in PAPER.items() for a, v in vals.items()
           if abs(q.loc[a, c] - v) > 5e-4 + 1e-12]
    bad += [("ties", a, q.loc[a, "ties"], v) for a, v in PAPER_TIES.items()
            if abs(q.loc[a, "ties"] - v) > 0.05 + 1e-12]
    if bad:
        sys.exit("does not match the paper: " + "; ".join(f"{c}[{a}] = {x:.4f} (paper {v})" for c, a, x, v in bad))


def main():
    os.chdir(HERE)  # DATA_DIR is relative to pipeline/<stage>/
    ties = tie_rates()
    parts = [aggregate(j, ties) for j in JUDGES]
    summary, pairwise, src = (pd.concat(p, ignore_index=True) for p in zip(*parts))
    check_paper(summary)
    FINAL_DIR.mkdir(exist_ok=True)
    for name, df in (("arm_summary", summary), ("pairwise_pvalues", pairwise), ("src_summary", src)):
        df.to_csv(FINAL_DIR / f"{name}.csv", index=False)
        print(f"wrote {FINAL_DIR / name}.csv {df.shape}")


if __name__ == "__main__":
    main()
