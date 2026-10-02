# kb_mapper

Normalises free text onto the labels of a knowledge base graph (`kb_graph`). The text can be a
skill written in a CV or a job ad, or a job title. The mapper returns one label of the graph, or
says that none fits.

## How it works

```
map(s, context):
    if normalise(s) in alias_index: return EXACT(label)
    C = top-k labels by cosine(embed(s), embed(label))
    c = LLM picks one of C ∪ {NO_MATCH}, given s, context, descriptions(C)
    return NO_MATCH if c = NO_MATCH else LLM_RERANK(c)
```

- `normalise` strips accents, case and separators: `"Atención-al/Cliente"` becomes
  `"atencion al cliente"`. `alias_index` holds every label of the pool plus its alternative
  labels.
- `embed` is a sentence transformer, `abd1987/esco-context-skill-extraction`, with `k = 12`.
- The LLM is `qwen3:8b`, run through Ollama at temperature 0. It sees each candidate with the
  first 240 characters of its description, and must answer with one of them or `NO_MATCH`. Its
  reply is constrained to that JSON schema.
- Results are cached on `(normalise(s), context)`.

## Input

- `labels`: the pool of labels to map onto, for example every skill at level ≥ 4
  (`skill_pool_uris`).
- `processed_dir`: the folder `kb_graph` reads (see its README). The mapper uses the `label`,
  `altLabels` and `description` columns of `skills.csv` / `occupations.csv`, and the
  `*_alt_labels.csv` files. Skill groups (`broad_skill`) contribute no descriptions.
- `node`: `"skill"` or `"occupation"`.
- `kb`: the name the LLM prompt uses for the knowledge base.
- Runtime: an Ollama server with `qwen3:8b`. The mapper calls it only for strings that have no
  exact match.

## Usage

```python
from pathlib import Path
from kb_graph import KBGraph
from kb_mapper import KBMapper, skill_pool_uris

g = KBGraph.load(Path("cache/occSkill_graph.pkl"))
pool = sorted(g.get_label(u) for u in skill_pool_uris(g, min_level=4))
mapper = KBMapper(pool, node="skill", processed_dir=Path("sources/ESCO/processed"), kb="ESCO")
len(pool)
# 13866
```

```python
mapper.map_skill("  python ")
# {'kb_label': 'Python (computer programming)', 'semantic_score': 1.0, 'semantic_rank': 1, 'candidate_list': '["Python (computer programming)"]', 'decision_source': 'EXACT', 'confidence': 'HIGH', 'mapping_reason': 'Exact match against an ESCO skill label or alias.'}
```

```python
mapper.map_skill("Manage-Employees")
# {'kb_label': 'manage staff', 'semantic_score': 1.0, 'semantic_rank': 1, 'candidate_list': '["manage staff"]', 'decision_source': 'EXACT', 'confidence': 'HIGH', 'mapping_reason': 'Exact match against an ESCO skill label or alias.'}
```

A string with no exact match goes through retrieval and the LLM. This result comes from a real
run of the pipeline on a job ad, with `context` set to the ad's job title:

```python
mapper.map_skill("Android SDK", context=...)
# {'kb_label': 'mobile device software frameworks', 'semantic_score': 0.8044, 'semantic_rank': 2,
#    'candidate_list': '["Android (mobile operating systems)", "mobile device software frameworks", "Xcode", "iOS", ...]',
#    'decision_source': 'LLM_RERANK', 'confidence': 'HIGH',
#    'mapping_reason': 'The Android SDK is a framework that provides APIs for developing applications on mobile devices, which aligns closely with the description of mobile device software frameworks.'}
```

Every result has the same keys:

- `kb_label`: the chosen label, or `None` for `NO_MATCH`;
- `semantic_score` and `semantic_rank`: the retrieval score and rank of the chosen candidate;
- `candidate_list`: the candidates, as JSON;
- `decision_source`: `EXACT`, `LLM_RERANK` or `NO_MATCH`;
- `confidence` and `mapping_reason`: the LLM's answer.
