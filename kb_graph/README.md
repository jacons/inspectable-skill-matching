# kb_graph

One graph for a labour-market knowledge base (ESCO, O*NET, or any taxonomy with the same shape):
occupations and skills are the nodes; their taxonomies and the occupation–skill relations are the
edges.

## Input

A folder with five CSV files. Ids are strings. `altLabels` is a list of ids into the
`*_alt_labels.csv` files, which only the mapper reads. `level` and `leaf` come from
`taxonomy_levels`.

| file | columns | one row is |
|---|---|---|
| `occupations.csv` | `occ_id, label, altLabels, code, level, leaf, description` | an occupation or an occupation group |
| `skills.csv` | `skill_id, skill_type, label, reuseLevel, altLabels, level, leaf, description` | a skill or a skill group |
| `occupation_tax.csv` | `conceptUri, broaderUri` | a (child, parent) link between occupations |
| `skill_tax.csv` | `conceptUri, broaderUri` | a (child, parent) link between skills |
| `occ_skill_rel.csv` | `occupationUri, relationType, skillUri` | an occupation that requires a skill (`essential` / `optional`) |

`skill_type` is free text. The graph gives meaning to two of its values: `broad_skill` marks a
group of skills, and `language` marks a language skill (see `is_language`).

The first rows of the ESCO files (descriptions cut):

```text
occ_id,label,altLabels,code,leaf,level,description
00030d09-2b3a-4efd-87cc-c4ea39d27c34,technical director,"[0, 1, 2, 3, 4, 5]",2654.1.7,True,6,"Technical directors realise the artistic visions of ..."

skill_id,skill_type,label,reuseLevel,altLabels,description,level,leaf
0005c151-5b5a-4a66-8aac-60e734beb1ab,skill/competence,manage musical staff,sector-specific,"[0, 1, 2, 3]","Assign and manage staff tasks in areas such as ...",5,True

conceptUri,broaderUri          (occupation_tax.csv)
C011,C01

conceptUri,broaderUri          (skill_tax.csv)
000,00

occupationUri,relationType,skillUri
00030d09-2b3a-4efd-87cc-c4ea39d27c34,essential,fed5b267-73fa-461d-9f69-827c78beb39d
```

`level` is 1 at a root, then the shortest distance from a root. `leaf` is true for a node with
no narrower concept:

```python
from kb_graph import taxonomy_levels
level, leaf = taxonomy_levels(pd.read_csv("processed/skill_tax.csv"))
```

The notebooks in `sources/ESCO/` and `sources/ONET/` build these files from the official ESCO
v1.2.1 and O*NET 31.0 downloads.

## Usage

```python
from pathlib import Path
from kb_graph import KBGraph

g = KBGraph.from_processed(Path("sources/ESCO/processed"))   # build once from the CSV files ...
g.save(Path("cache/occSkill_graph.pkl"))
g = KBGraph.load(Path("cache/occSkill_graph.pkl"))           # ... then load the cache
sum(1 for _ in g.nodes({"type": "occupation"})), sum(1 for _ in g.nodes({"type": "skill"}))
# (3658, 14579)
```

```python
uri = g.get_uri("Python (computer programming)")
uri, g.get_label(uri), g.get_level(uri), g.node_info(uri)["skill_type"]
# ('ccd0a1d9-afda-43d9-b901-96344886e14d', 'Python (computer programming)', 5, 'knowledge')
```

When several nodes share a label (a group and one of its members, say), `get_uri` returns the
shallowest by default (`reduce="Top"`) and the deepest with `reduce="Down"`.

```python
sorted(g.get_label(o) for o in g.neighbors(uri, {"type": "occupation"}))[:5]
# ['3D modeller', 'ICT application configurator', 'ICT application developer', 'ICT network administrator', 'ICT network engineer']
```

The graph offers two measures of how close two nodes of the same type are:

- `how="hop"` counts the edges on the shortest path: 0 is the same node, `inf` means no path.
  `intermediate_properties={"type": "skill"}` keeps the path inside the skill taxonomy.
- `how="neighbor"` is the Jaccard overlap, in [0, 1], of the occupations that require the two
  skills (for two occupations, of the skills they require).

```python
a = g.get_uri("geriatrics", reduce="Down")
b = g.get_uri("orthopaedics", reduce="Down")
g.similarity(a, b, how="hop", intermediate_properties={"type": "skill"})
# 2.0
```

```python
g.path(a, b, labeled=True, intermediate_properties={"type": "skill"})
# ['geriatrics', 'medicine', 'orthopaedics']
```

```python
g.similarity(a, b, how="neighbor")
# 0.32
```

Hierarchies and languages:

```python
g.ancestor_at_level(g.get_uri("market vendor"), level=2, labeled=True)
# 'Sales workers'
```

```python
g.is_language(label="understand spoken Italian"), g.generalize_lang(label="understand spoken Italian")
# (True, 'Italian')
```

```python
onet = KBGraph.load(Path("cache/onet_occSkill_graph.pkl"))
onet.get_label("15-1252.00"), onet.ancestor_at_level("15-1252.00", level=1, labeled=True)
# ('Software Developers', 'Computer and Mathematical Occupations')
```
