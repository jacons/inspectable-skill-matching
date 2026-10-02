# Inspectable Skill Matching for Accountable Candidate Ranking

Code and data for the paper *Inspectable Skill Matching for Accountable Candidate Ranking*. The
repository builds job-CV fitness features with skill-matching functions grounded in two
knowledge bases (ESCO and O*NET), trains a LightGBM ranker on each feature set, and compares the
rankings against LLM relevance labels.

The supplementary material is in [`supplemental_materials.pdf`](supplemental_materials.pdf).

## Repository layout

```text
.
├── kb_graph/                    # a knowledge base as a graph: occupations, skills, taxonomies,
│                                #   hop and neighbor similarity (see kb_graph/README.md)
├── kb_mapper/                   # maps free-text skills and job titles to KB labels:
│                                #   exact match, embedding retrieval, LLM rerank (see kb_mapper/README.md)
├── feature_engineering/         # skill fitness per method and KB; education, experience and
│                                #   language matches (see feature_engineering/README.md)
├── pipeline/
│   ├── 0_align_datasets/
│   │   ├── curricula/           # CVs: preprocessing, translation, skill mapping, export
│   │   └── job_offer/           # job offers: preprocessing, translation, title/skill/language mapping, export
│   ├── 1_ground_truth/          # applications (job + pool of CVs) and LLM relevance labels
│   ├── 2_fitness/               # fitness matrix of every job-CV pair
│   ├── 3_ranking/               # LightGBM ranking arms, 5-fold CV, placebo replicates
│   ├── 4_evaluation/            # gain and rank change per judge; evaluation_all_judges.py
│   │                            #   aggregates all judges, 8_1_final_results.ipynb makes tables and figures
│   └── utils/                   # shared helpers; kb_explanation.py builds the worked example
├── datasets/findhr/             # ids and scores only, no CV or job text
│   ├── applications/            # the applications: job id and pool of CV ids
│   ├── judges/<judge>/          # LLM labels (llm_relevance.csv, ground_truth.csv)
│   │   └── results/             # per-query nDCG and candidate positions of every arm
│   ├── skill_fitness.csv        # skill fitness of every pair, one column per (method, KB)
│   ├── fitness_matrix.csv       # the ranker's input: skill fitness plus fit_edu, fit_years, fit_lang
│   └── final/                   # results over all judges: CSV summaries, tables/ (LaTeX), imgs/ (PDF)
├── sources/
│   ├── ESCO/                    # ESCO v1.2.1 CSV download, processing notebooks, processed/ CSVs
│   ├── ONET/                    # O*NET 31.0 download, processing notebook, processed/ CSVs
│   └── build_kg_cache.py        # builds the two graph caches in cache/
├── kb_expl_examples/            # TikZ figures of the worked example and of the ESCO structure
├── supplemental_materials.pdf
├── requirements.txt
└── LICENSE
```

## What is not included

The paper is under double-blind review, and two datasets cannot be redistributed:

- the CVs, obtained from the FINDHR project, for privacy reasons;
- the job offers, provided by an HR company under an agreement. Publishing them would also
  identify the company and, through it, the authors.

For the same reasons the repository leaves out every file that holds or derives from their
text:

- the raw inputs in `pipeline/0_align_datasets/source/` (`offers_examples_10k.csv`,
  `anonymized_cvs.csv`);
- the `mapping/` folders of the stage-0 notebooks, including the hand-made
  `mapping/education_map.csv`;
- the stage-0 exports `job_offer_database.json` and `curricula_database.json`, and
  `datasets/findhr/source/` (the aligned job offers and CVs the experiment reads). Two more folders are left out because you can rebuild them: `cache/` (the
graph caches) and `datasets/findhr/judges/*/model/` (the trained rankers).

What you can run with the published files:

| part | runs | needs |
|---|---|---|
| `kb_graph`, `kb_mapper` (exact match), `feature_engineering` | yes | `cache/`; the sentence transformers download from Hugging Face on first use |
| `kb_mapper` (LLM rerank) | yes | an Ollama server with `qwen3:8b` |
| `pipeline/4_evaluation/evaluation_all_judges.py`, `8_1_final_results.ipynb` | yes | `datasets/findhr/` |
| `pipeline/3_ranking/6_ranking.ipynb`, then `7_gain_evaluation.ipynb` | yes | `datasets/findhr/`; 7 reads the models 6 writes to `judges/<judge>/model/` |
| `pipeline/utils/kb_explanation.py`: `explain_pair`, `explain_job_skill` | yes | `cache/` |
| `pipeline/utils/kb_explanation.py`: `main()` | no | the private job offers and CVs |
| `pipeline/0_align_datasets`, `1_ground_truth`, `2_fitness` | no | the private job offers and CVs |

You can still read the notebooks of stages 0-2: their code and comments describe every step.

## Setup

Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sources/build_kg_cache.py      # writes cache/occSkill_graph.pkl and cache/onet_occSkill_graph.pkl
```

The notebooks add the repository to `sys.path` with a relative path, so open each one from its
own folder (the Jupyter default), for example after `jupyter lab` from the repository root.

Only stages 0-1 and the `kb_mapper` LLM rerank call an LLM. They need an
[Ollama](https://ollama.com) server with the models in `pipeline/utils/ranking_utils.py`
(`JUDGES`) and `qwen3:8b`; set `OLLAMA_HOST` if the server is not local.

## Reproducing the results

```bash
python pipeline/4_evaluation/evaluation_all_judges.py   # about 6 minutes on a laptop
```

It reads the per-query results of the three judges, writes `datasets/findhr/final/*.csv`, and
exits if the values of the paper's arm summary table are not reproduced. Then run
`pipeline/4_evaluation/8_1_final_results.ipynb`: it writes every table to
`datasets/findhr/final/tables/` and every figure to `datasets/findhr/final/imgs/`. With
`pdflatex` and `pdftoppm` on the PATH the notebook also shows each table rendered; without them
it prints the LaTeX source.

To retrain the rankers, run `pipeline/3_ranking/6_ranking.ipynb` and then
`pipeline/4_evaluation/7_gain_evaluation.ipynb` once per judge (set `JUDGE` in the first cell).
The 80 placebo replicates make this slow.

## Examples

### kb_graph

```python
from pathlib import Path
from kb_graph import KBGraph

g = KBGraph.load(Path("cache/occSkill_graph.pkl"))
a = g.get_uri("geriatrics", reduce="Down")
b = g.get_uri("orthopaedics", reduce="Down")

g.similarity(a, b, how="hop", intermediate_properties={"type": "skill"})
# 2.0
g.path(a, b, labeled=True, intermediate_properties={"type": "skill"})
# ['geriatrics', 'medicine', 'orthopaedics']
g.similarity(a, b, how="neighbor")
# 0.32
```

`how="hop"` counts the edges between two skills inside the skill taxonomy; `how="neighbor"` is
the Jaccard overlap of the occupations that require them. The same calls work on O*NET
(`cache/onet_occSkill_graph.pkl`). [`kb_graph/README.md`](kb_graph/README.md) documents the
input files and the rest of the API.

### kb_explanation

`explain_pair` returns the numbers behind the two KB similarities for one job skill and one CV
skill, with the evidence: the shared occupations and the taxonomy path.

```python
from pipeline.utils.kb_explanation import explain_pair, explain_job_skill

e = explain_pair(g, "geriatrics", "orthopaedics")
len(e["shared"]), len(e["union"]), e["neighbor"]
# (8, 25, 0.32)
sorted(e["shared"])[:3]
# ['chiropractor', 'healthcare specialist lecturer', 'medicine lecturer']
e["distance"], round(e["hop"], 4), e["path"]
# (2.0, 0.3679, ['geriatrics', 'medicine', 'orthopaedics'])

explain_pair(g, "numerology", "orthopaedics")["distance"]
# 8.0, beyond the 6-hop cut-off, so its hop score is 0
```

`explain_job_skill(g, job_skill, cv_skills)` picks, for one job skill, the closest CV skill under
each similarity. `python -m pipeline.utils.kb_explanation` builds the worked example of the
supplementary material (`kb_expl_examples/expl1-4`), but it needs the private job and CV records.

### Further reading

- [`kb_graph/README.md`](kb_graph/README.md): input format, similarities, hierarchies, languages.
- [`kb_mapper/README.md`](kb_mapper/README.md): the mapping cascade and its output.
- [`feature_engineering/README.md`](feature_engineering/README.md): skill fitness methods and the fitness matrix.

## Data sources and licenses

- ESCO v1.2.1, European Commission, reused under the Commission's reuse policy
  (Decision 2011/833/EU). `sources/ESCO/` holds the original download.
- O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `sources/ONET/` holds the original
  download and the O*NET-SOC 2019 structure file.
- The code is released under the MIT license ([`LICENSE`](LICENSE)).
