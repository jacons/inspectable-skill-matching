"""fit_edu, fit_years, fit_lang of (job, CV) pairs, joined with the skill fitness columns."""
import re

import pandas as pd


def first_number(text) -> int:
    """First integer in a text: "level 6" -> 6, "5 - 9 years" -> 5, "Not specified" -> 0."""
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else 0


def overlap(required, offered) -> float:
    """Share of the required items that are offered (1.0 when nothing is required)."""
    return len(set(required) & set(offered)) / len(required) if len(required) else 1.0


def build_fitness_matrix(pairs: pd.DataFrame, jobs: pd.DataFrame, cvs: pd.DataFrame,
                         skill_fitness: pd.DataFrame) -> pd.DataFrame:
    """`pairs`: id_j, id_c. `jobs` / `cvs`: id_j / id_c plus edu_eqf, years_experience and
    languages suffixed _j / _c. `skill_fitness`: id_j, id_c and the fit_skills_* columns."""
    df = pairs.merge(jobs, on="id_j").merge(cvs, on="id_c")
    fit = df[["id_j", "id_c"]].copy()
    fit["fit_edu"] = (df["edu_eqf_j"].map(first_number) <= df["edu_eqf_c"].map(first_number)).astype(float)
    fit["fit_years"] = (df["years_experience_j"].map(first_number)
                        <= df["years_experience_c"].map(first_number)).astype(float)
    fit["fit_lang"] = [overlap(j, c) for j, c in zip(df["languages_j"], df["languages_c"])]
    return fit.merge(skill_fitness, on=["id_j", "id_c"])
