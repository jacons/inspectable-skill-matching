"""Resumable LLM loops of the stage-0 notebooks: process the pending rows and append the
results in chunks, so an interrupted run restarts where it stopped. Rows are matched by key
(`id_j` for job offers, `id_c` for CVs)."""
import csv
import sys
from pathlib import Path
from typing import Callable, List

import pandas as pd
from pydantic import create_model
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from kb_mapper import chat_json

# (path, columns) of an output CSV; the first column is always the key
Output = tuple[str | Path, list[str]]


def pending(src: pd.DataFrame, *outputs: Output) -> pd.DataFrame:
    """Create the missing output CSVs and return the `src` rows whose key is not in the first one.

    The key is the first column of every output.
    """
    key = outputs[0][1][0]
    assert src[key].is_unique, f"{key} must be a unique key"
    for path, columns in outputs:
        assert columns[0] == key, f"{path}: the first column must be {key}, not {columns[0]}"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)

    done = set(pd.read_csv(outputs[0][0])[key])
    todo = src[~src[key].isin(done)].reset_index(drop=True)
    print(f"already done: {len(done)} | to do: {len(todo)} / {len(src)}")
    return todo


def run_chunked(todo: pd.DataFrame, fn: Callable, *outputs: Output, chunk_size: int = 100) -> None:
    """Apply `fn` to every row of `todo` and append the results every `chunk_size` rows.

    `fn(row)` returns, for each output, a dict or a list of dicts (just the dict with one output).
    """
    for i in range(0, len(todo), chunk_size):
        subset = todo.iloc[i: i + chunk_size]
        batches: list[list[dict]] = [[] for _ in outputs]

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Chunk {i // chunk_size}", mininterval=2):
            result = fn(row)
            if len(outputs) == 1:
                result = (result,)
            assert len(result) == len(outputs), f"fn must return {len(outputs)} results"
            for batch, rows in zip(batches, result):
                batch.extend(rows if isinstance(rows, list) else [rows])

        for (path, columns), batch in zip(outputs, batches):
            if batch:
                # columns= keeps the header's order whatever the dict's key order
                pd.DataFrame(batch, columns=columns).to_csv(
                    path, mode="a", quoting=csv.QUOTE_ALL, index=False, header=False)


def _llm_translate_skills(skills_list: List[str], context: str, model: str = "qwen3:8b") -> List[str]:
    """Translate a list of skills into English in one LLM call; `context` (job title or
    category) only disambiguates. The output may not have the input's length."""
    format_output = create_model("SkillTranslationOutput", translations=(List[str], ...))
    system_prompt = (
        "You are an expert labor-market translator specialized in HR skills.\n"
        "Translate each Spanish skill in the list into English.\n"
        "Rules:\n"
        "- Translate EVERY item; keep the same order and the same count.\n"
        "- If an item is already in English, return it unchanged.\n"
        "- Keep each translation in skill register: short, nominal phrases, "
        "not full sentences or explanations.\n"
        "- Keep proper nouns, technologies and acronyms unchanged "
        "(e.g. Python, Excel, SAP, AutoCAD).\n"
        "- Use the job context ONLY to disambiguate polysemous terms "
        "(e.g. 'redes' -> 'networks' vs 'social media'); do not translate it "
        "and do not add skills that are not in the list.\n"
        "Return only valid JSON matching the provided schema."
    )
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(skills_list, 1))
    result = chat_json(
        output_model=format_output,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Job context (do not translate): {context}\nSkills:\n{numbered}"},
        ],
        task_name="skill translation",
    )
    return [t.strip() for t in result.translations]


def translate_skills_to_english(skills_list: List[str], context: str, model: str = "qwen3:8b") -> List[str]:
    """Translate `skills_list` from Spanish to English, one output per input, in order.

    The whole list goes in one call, retried once if the lengths differ; after that each skill
    is translated alone, and kept as is if that fails too.
    """
    if not skills_list:
        return []

    for _ in range(2):
        out = _llm_translate_skills(skills_list, context, model)
        if len(out) == len(skills_list):
            return out

    fallback = []
    for s in skills_list:
        one = _llm_translate_skills([s], context, model)
        fallback.append(one[0] if len(one) == 1 else s)
    return fallback
