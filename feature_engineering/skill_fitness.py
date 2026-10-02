"""Skill fitness of (job, CV) pairs: for every skill the job asks for, the best score against the
CV's skills, averaged over the job's skills. One column per method, see README.md."""
from functools import cache, cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from kb_graph import KBGraph

FIT_COLUMNS = {
    "sy": "fit_skills_sy",
    "neighbor": "fit_skills_kb_neighbor",
    "hop": "fit_skills_kb_hop",
    "transformer": "fit_skills_transformer",
    "transformer_raw": "fit_skills_transformer_raw",
}


def phi_distance(distance: float) -> float:
    """Score of a hop distance d: 1 at 0 hops, exp(-0.5 d) up to 6 hops, 0 beyond or with no path."""
    if distance == 0:
        return 1.0
    return float(np.exp(-0.5 * distance)) if distance <= 6 else 0.0


def score_column(fit_method: str, kb: str | None = None) -> str:
    """Output column of a method; the KB-dependent ones get the KB as suffix."""
    column = FIT_COLUMNS[fit_method]
    return column if kb is None or fit_method == "transformer_raw" else f"{column}_{kb}"


class SkillFitnessCalculator:
    def __init__(self, graph: Path, transformer: str):
        self.graph = KBGraph.load(graph)
        self.transformer = transformer
        self.match = cache(self._match)
        self.embed = cache(self._embed)

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.transformer, device="cpu")

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    @staticmethod
    def exact_fit(job_skills, candidate_skills) -> float:
        """Share of the job's skills the CV lists verbatim (1.0 when the job asks for none)."""
        job, candidate = set(job_skills), set(candidate_skills)
        return len(job & candidate) / len(job) if job else 1.0

    def score_dataframe(self, df: pd.DataFrame, fit_methods, skill_columns=("skills_j", "skills_c"),
                        kb: str | None = None) -> pd.DataFrame:
        """One score column per method for every row of `df`; `skill_columns` name the job and CV lists."""
        pairs = [(frozenset(j), frozenset(c)) for j, c in df[list(skill_columns)].itertuples(index=False, name=None)]
        return pd.DataFrame({
            score_column(method, kb): [self.score_pair(j, c, method) for j, c in tqdm(pairs, desc=f"{method} skill fit")]
            for method in fit_methods
        }, index=df.index)

    def score_pair(self, job_skills, candidate_skills, fit_method: str) -> float:
        if fit_method == "sy":
            return self.exact_fit(job_skills, candidate_skills)
        if not job_skills:
            return 1.0
        candidate_skills = frozenset(candidate_skills)
        return float(np.mean([self.best_match(s, candidate_skills, fit_method) for s in sorted(job_skills)]))

    def best_match(self, job_skill: str, candidate_skills: frozenset, fit_method: str) -> float:
        if job_skill in candidate_skills:
            return 1.0
        if not candidate_skills:
            return 0.0
        return max(self.match(job_skill, c, fit_method) for c in candidate_skills)

    def _match(self, a: str, b: str, fit_method: str) -> float:
        if a == b:
            return 1.0
        if fit_method == "neighbor":
            return self.graph.similarity(a, b, how="neighbor")
        if fit_method == "hop":
            return phi_distance(self.graph.similarity(a, b, how="hop", intermediate_properties={"type": "skill"}))
        if fit_method == "transformer":
            a, b = self.graph.get_label(a), self.graph.get_label(b)
        return (float(self.embed(a) @ self.embed(b)) + 1.0) / 2.0
