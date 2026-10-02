# feature_engineering

Turns a (job, CV) pair into the features the ranker reads:

- one skill fitness score per method and knowledge base: `skill_fitness.py`, written to
  `datasets/findhr/skill_fitness.csv`;
- three non-skill matches (education, experience, languages): `fitness_matrix.py`, written to
  `datasets/findhr/fitness_matrix.csv`.

## Skill fitness

Input: for each pair, the job's skills and the CV's skills as lists of node ids of a `kb_graph`
graph. These are the labels `kb_mapper` produced, turned into ids with
`get_uri(label, reduce="Down")`. The `transformer_raw` method reads the raw skill strings instead.

Every method is job-centric. For each skill the job asks for, it takes the best match
among the CV's skills (1 when the CV lists the same skill), then averages over the job's skills.
A job with no skills scores 1.

| method | column | match between job skill *a* and CV skill *b* |
|---|---|---|
| `sy` | `fit_skills_sy_<kb>` | share of the job's skills the CV lists verbatim (no best-match step) |
| `neighbor` | `fit_skills_kb_neighbor_<kb>` | Jaccard overlap of the occupations requiring *a* and *b* |
| `hop` | `fit_skills_kb_hop_<kb>` | `phi(d)`, where *d* is the number of hops between *a* and *b* inside the skill taxonomy: 1 at 0 hops, `exp(-0.5 d)` up to 6 hops, 0 beyond 6 hops or with no path |
| `transformer` | `fit_skills_transformer_<kb>` | `(cos + 1) / 2` between the embeddings of the two KB labels |
| `transformer_raw` | `fit_skills_transformer_raw` | the same cosine, on the raw strings (no KB) |

`fit_skills_raw` is the `sy` rule applied to the raw strings. `4_skill_fitness.ipynb` computes it
with `SkillFitnessCalculator.exact_fit`.

```python
from pathlib import Path
import pandas as pd
from feature_engineering.skill_fitness import SkillFitnessCalculator, phi_distance

calc = SkillFitnessCalculator(graph=Path("cache/occSkill_graph.pkl"),
                              transformer="sentence-transformers/all-MiniLM-L6-v2")
job = ["geriatrics", "orthopaedics", "numerology"]          # labels of the KB, as mapped by kb_mapper
cv = ["orthopaedics", "physiotherapy", "rehabilitation"]
to_uris = lambda labels: [calc.graph.get_uri(l, reduce="Down") for l in labels]
pair = pd.DataFrame({"skills_j": [to_uris(job)], "skills_c": [to_uris(cv)]})
calc.score_dataframe(pair, fit_methods=("sy", "neighbor", "hop", "transformer"), kb="esco").round(4).T
#                                   0
#   fit_skills_sy_esco           0.3333
#   fit_skills_kb_neighbor_esco  0.4400
#   fit_skills_kb_hop_esco       0.4560
#   fit_skills_transformer_esco  0.7650
```

```python
raw = pd.DataFrame({"raw_skills_j": [["elderly care", "orthopaedic nursing"]],
                    "raw_skills_c": [["physiotherapy", "patient rehabilitation"]]})
calc.score_dataframe(raw, fit_methods=("transformer_raw",),
                     skill_columns=("raw_skills_j", "raw_skills_c")).round(4).T
#                                  0
#   fit_skills_transformer_raw  0.7166
```

```python
[round(phi_distance(d), 4) for d in (0, 1, 2, 6, 7, float("inf"))]
# [1.0, 0.6065, 0.3679, 0.0498, 0.0, 0.0]
```

`pipeline/2_fitness/4_skill_fitness.ipynb` runs this on every pair of the applications, once per
knowledge base (`kb="esco"`, `kb="onet"`). It writes `skill_fitness.csv`: `id_j, id_c` and one
column per (method, KB).

## Fitness matrix

`build_fitness_matrix` adds three matches and joins the skill columns:

- `fit_edu`: 1 when the CV's EQF level is at least the job's;
- `fit_years`: 1 when the CV's minimum years of experience are at least the job's minimum;
- `fit_lang`: the share of the job's languages the CV speaks.

```python
from feature_engineering.fitness_matrix import build_fitness_matrix
pairs = pd.DataFrame({"id_j": [1], "id_c": [7]})
jobs = pd.DataFrame({"id_j": [1], "edu_eqf_j": ["level 6"], "years_experience_j": ["1 - 2 years"],
                     "languages_j": [["English", "Spanish"]]})
cvs = pd.DataFrame({"id_c": [7], "edu_eqf_c": ["level 7"], "years_experience_c": ["0 - 4 years"],
                    "languages_c": [["Spanish"]]})
skills = pd.DataFrame({"id_j": [1], "id_c": [7], "fit_skills_kb_hop_esco": [0.51]})
build_fitness_matrix(pairs, jobs, cvs, skills)
#    id_j  id_c  fit_edu  fit_years  fit_lang  fit_skills_kb_hop_esco
#   0     1     7      1.0        0.0       0.5                    0.51
```
