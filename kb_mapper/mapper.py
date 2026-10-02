import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

import ollama
import pandas as pd
from pydantic import BaseModel, create_model
from sentence_transformers import SentenceTransformer, util

MAX_TOKENS = 2048
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


def skill_pool_uris(graph, min_level: int) -> set[str]:
    """Ids of the skills at `min_level` or deeper, skill groups (broad_skill) excluded."""
    return {uri for uri, attrs in graph.nodes({"type": "skill"})
            if attrs["skill_type"] != "broad_skill" and attrs["level"] >= min_level}


def extract_json_object(text: str) -> str:
    """The {...} span of a model reply, without the prose or code fences around it."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text.strip()


def chat_json(*, output_model: type[BaseModel], messages: list[dict], task_name: str, model: str = "qwen3:8b"):
    """Ask the LLM for a reply constrained to `output_model`'s JSON schema, at temperature 0.
    A reply cut at MAX_TOKENS (a repetition loop) is retried once with repeat_penalty 1.1."""
    schema = output_model.model_json_schema()
    options = {"temperature": 0, "num_predict": MAX_TOKENS}
    response = ollama.chat(model=model, messages=messages, think=False, format=schema, options=options)
    if response.done_reason == "length":
        response = ollama.chat(model=model, messages=messages, think=False, format=schema,
                               options={**options, "repeat_penalty": 1.1})
    raw = response.message.content or ""
    try:
        if response.done_reason == "length":
            raise ValueError(f"reply truncated at {MAX_TOKENS} tokens")
        return output_model.model_validate_json(extract_json_object(raw))
    except Exception as exc:
        raise RuntimeError(f"{model} gave no valid JSON for {task_name}: {exc}. Reply: {raw[:500]!r}") from exc


def normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalise_lookup_text(value: str) -> str:
    """Key for the exact match: no accents, casefolded, separators as spaces.
    "  Atención-al/Cliente " -> "atencion al cliente"."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    return normalise_space(re.sub(r"[\s_/\-]+", " ", text))


def load_alias_lookup(node: Literal["skill", "occupation"], processed_dir: Path) -> dict[str, set[str]]:
    """For every label, the set of the label and its alternative labels."""
    nodes_file, alias_file = {"skill": ("skills.csv", "skills_alt_labels.csv"),
                              "occupation": ("occupations.csv", "occ_alt_labels.csv")}[node]
    nodes = pd.read_csv(processed_dir / nodes_file, converters={"altLabels": ast.literal_eval})
    aliases = pd.read_csv(processed_dir / alias_file, dtype={"alt_label_id": "string"})
    id_to_alias = {str(i): a for i, a in zip(aliases["alt_label_id"], aliases["alt_label"]) if isinstance(a, str) and a}
    lookup = {}
    for label, alt_ids in zip(nodes["label"], nodes["altLabels"]):
        label = str(label).strip()
        if label:
            lookup.setdefault(label, {label}).update(id_to_alias[str(i)] for i in alt_ids if str(i) in id_to_alias)
    return lookup


def load_descriptions(node: Literal["skill", "occupation"], processed_dir: Path, max_chars: int = 240) -> dict[str, str]:
    """The first `max_chars` characters of the description of every label; skill groups are left out."""
    df = pd.read_csv(processed_dir / f"{node}s.csv").fillna({"description": ""})
    if node == "skill":
        df = df[df["skill_type"] != "broad_skill"]
    return {label: description[:max_chars] for label, description in zip(df["label"], df["description"])}


class KBMapper:
    """Map free text (a skill, a job title) onto one label of a fixed pool of KB labels:
    exact match on labels and aliases, else the `top_k` closest labels by embedding reranked by an LLM."""

    def __init__(self, labels, node: Literal["skill", "occupation"], processed_dir: Path, kb: str = "ESCO",
                 model: str = "qwen3:8b", top_k: int = 12,
                 transformer: str = "abd1987/esco-context-skill-extraction"):
        self.node, self.kb, self.model, self.top_k = node, kb, model, top_k
        self.labels = sorted(labels)
        self.embedder = SentenceTransformer(transformer)
        self.embeddings = self.embedder.encode(self.labels, convert_to_tensor=True)
        self.descriptions = load_descriptions(node, processed_dir)
        aliases = load_alias_lookup(node, processed_dir)
        self.exact_lookup = {}
        for label in self.labels:
            for alias in aliases.get(label, {label}):
                if key := normalise_lookup_text(alias):
                    self.exact_lookup.setdefault(key, label)
        self._cache = {}

    def map_skill(self, skill: str, context: str = "") -> dict:
        """Map one string; `context` (e.g. the job title) is shown to the LLM and is part of the cache key."""
        key = (normalise_lookup_text(skill), context)
        if key not in self._cache:
            self._cache[key] = self._map(str(skill), context)
        return dict(self._cache[key])

    def _map(self, skill: str, context: str) -> dict:
        if exact := self.exact_lookup.get(normalise_lookup_text(skill)):
            return self._row(exact, 1.0, 1, [exact], "EXACT", "HIGH",
                             f"Exact match against an {self.kb} {self.node} label or alias.")
        embedding = self.embedder.encode(skill, convert_to_tensor=True)
        hits = util.semantic_search(embedding, self.embeddings, top_k=self.top_k)[0]
        candidates = [{"label": self.labels[hit["corpus_id"]], "score": float(hit["score"]), "rank": rank + 1}
                      for rank, hit in enumerate(hits)]
        labels = [c["label"] for c in candidates]
        picked, confidence, reason = self._rerank(skill, context, candidates)
        if picked == "NO_MATCH":
            return self._row(None, 0.0, 0, labels, "NO_MATCH", confidence, reason)
        chosen = next(c for c in candidates if c["label"] == picked)
        return self._row(picked, chosen["score"], chosen["rank"], labels, "LLM_RERANK", confidence, reason)

    def _rerank(self, skill: str, context: str, candidates: list[dict]):
        output_model = create_model(
            f"{self.node}MapperOutput",
            esco_skill=(Literal[tuple([c["label"] for c in candidates] + ["NO_MATCH"])], ...),
            confidence=(Confidence, ...),
            mapping_reason=(str, ...),
        )
        system_prompt = (
            f"You are an {self.kb} {self.node} mapper.\n"
            f"Pick the single candidate whose meaning best matches the input {self.node}.\n"
            f"If no candidate is a reasonable semantic match, answer: NO_MATCH.\n"
            "Return only valid JSON matching the schema."
        )
        payload = {
            "skill": skill,
            "context": context,
            "candidates": [{"esco_skill": c["label"], "description": self.descriptions.get(c["label"], "")}
                           for c in candidates],
        }
        result = chat_json(
            output_model=output_model,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            task_name=f"skill reranking for {skill!r}",
        )
        return result.esco_skill, result.confidence, normalise_space(result.mapping_reason)

    @staticmethod
    def _row(label, score: float, rank: int, candidates: list[str], source: str, confidence: str, reason: str) -> dict:
        return {
            "kb_label": label,
            "semantic_score": score,
            "semantic_rank": rank,
            "candidate_list": json.dumps(candidates, ensure_ascii=False),
            "decision_source": source,
            "confidence": confidence,
            "mapping_reason": reason,
        }
