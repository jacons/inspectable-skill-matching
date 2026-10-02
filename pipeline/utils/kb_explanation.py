"""Explain the KB similarities on one job-CV pair (job 3031, CV 447).

The two skill sets share no ESCO entity, so exact matching scores 0. The neighbor similarity
still connects them through occupations that require both skills, and the hop similarity
through the skill taxonomy.

main() prints the walk-through, writes the TikZ figures expl1-4 and kb_explanation_table.tex to
kb_expl_examples/, and checks its scores against datasets/findhr/skill_fitness.csv. It reads the
job and CV records from datasets/findhr/source/, which is not distributed; explain_pair and
explain_job_skill need only the graph.

Run: python -m pipeline.utils.kb_explanation
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kb_graph import KBGraph  # noqa: E402
from feature_engineering.skill_fitness import phi_distance  # noqa: E402

ID_J, ID_C = 3031, 447
DATASET = ROOT / "datasets/findhr"
LATEX = ROOT / "kb_expl_examples"

# Same palette as kb_expl_examples/example_esco.tex.
COLORS = {
    "occLeaf": "FCE3C1",  # occupation
    "skLeaf": "CFE6F7",   # skill leaf
    "skGrp": "9FC9E8",    # skill group
    "shared": "E8617A",   # shared occupation or short path
    "far": "999999",      # path beyond the hop cut-off
}


def occupations(graph: KBGraph, uri: str) -> set[str]:
    """Labels of the occupations that require this skill."""
    return {graph.get_label(o) for o in graph.neighbors(uri, properties={"type": "occupation"})}


def explain_pair(graph: KBGraph, job_skill: str, cv_skill: str) -> dict:
    """Neighbor and hop scores of one (job skill, CV skill) pair, with the shared occupations and the path."""
    a, b = graph.get_uri(job_skill, reduce="Down"), graph.get_uri(cv_skill, reduce="Down")
    n_a, n_b = occupations(graph, a), occupations(graph, b)
    distance = graph.similarity(uri_a=a, uri_b=b, how="hop",
                                intermediate_properties={"type": "skill"})
    return {
        "job_skill": job_skill,
        "cv_skill": cv_skill,
        "occ_job": n_a,
        "occ_cv": n_b,
        "shared": n_a & n_b,
        "union": n_a | n_b,
        "neighbor": graph.similarity(uri_a=a, uri_b=b, how="neighbor"),
        "distance": distance,
        "hop": phi_distance(distance),
        "path": graph.path(a, b, labeled=True, intermediate_properties={"type": "skill"}),
    }


def explain_job_skill(graph: KBGraph, job_skill: str, cv_skills: list[str]) -> dict:
    """For one job skill, the closest CV skill under each similarity (they can differ)."""
    pairs = [explain_pair(graph, job_skill, cv) for cv in cv_skills]
    return {
        "job_skill": job_skill,
        "neighbor": max(pairs, key=lambda p: p["neighbor"]),
        "hop": min(pairs, key=lambda p: p["distance"]),
    }


def describe(rows: list[dict], job: dict, cv: dict, totals: dict) -> str:
    """Text report: one paragraph per job skill."""
    out = [
        f'Job {ID_J} "{job["raw_occupation"]}"  ({job["esco_occupation"]})',
        f"  requires : {', '.join(job['esco_skills'])}",
        f'CV {ID_C}  ({cv["esco_job_category"]})',
        f"  offers   : {', '.join(cv['esco_skills'])}",
        "",
        "Exact ESCO matching (SY_esco) scores this pair 0.000: the two sets do not share",
        "one entity. The KB scores:",
        "",
    ]
    for row in rows:
        nb, hp = row["neighbor"], row["hop"]
        out.append(f'REQUIREMENT: "{row["job_skill"]}"')
        if nb["shared"]:
            shown = sorted(nb["shared"])
            head = ", ".join(shown[:4])
            rest = f" (+{len(shown) - 4} more)" if len(shown) > 4 else ""
            out += [
                f'  neighbour  -> "{nb["cv_skill"]}"   {nb["neighbor"]:.3f}',
                f"     {len(nb['shared'])} of the {len(nb['union'])} occupations that require either concept require BOTH:",
                f"     {head}{rest}.",
            ]
        else:
            out += [
                f"  neighbour  -> nothing   0.000",
                f"     no occupation requires this and any of the candidate's skills at once.",
                f"     ESCO links it to {len(nb['occ_job'])} occupation(s): {', '.join(sorted(nb['occ_job']))}.",
            ]
        out += [
            f'  taxonomy   -> "{hp["cv_skill"]}"   d={hp["distance"]:.0f} hops, phi={hp["hop"]:.3f}',
            f"     {' -> '.join(hp['path'])}",
        ]
        if hp["distance"] > 6:
            out.append("     beyond the 6-hop cut-off: scored 0.")
        out.append("")
    out += [
        "AGGREGATE (mean over the required skills, job-centric):",
        f"  SY_esco     {totals['sy']:.4f}",
        f"  KB_neighbor {totals['neighbor']:.4f}",
        f"  KB_hop      {totals['hop']:.4f}",
    ]
    return "\n".join(out)


def markdown_table(rows: list[dict], totals: dict) -> str:
    """The same numbers as a markdown table."""
    head = ("| required skill | best candidate skill | shared occ. | Jaccard | hops | phi |\n"
            "|---|---|---|---|---|---|")
    body = []
    for row in rows:
        nb, hp = row["neighbor"], row["hop"]
        match = nb["cv_skill"] if nb["shared"] else "--"
        body.append(
            f'| {row["job_skill"]} | {match} | {len(nb["shared"])}/{len(nb["union"])} '
            f'| {nb["neighbor"]:.3f} | {hp["distance"]:.0f} | {hp["hop"]:.3f} |'
        )
    body.append(f'| **mean** | | | **{totals["neighbor"]:.3f}** | | **{totals["hop"]:.3f}** |')
    return "\n".join([head, *body])


def latex_table(rows: list[dict], totals: dict) -> str:
    """The same numbers as a booktabs tabular."""
    def esc(text: str) -> str:
        return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")

    lines = [
        r"% generated by pipeline/utils/kb_explanation.py -- do not edit by hand",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Required skill (job) & Best candidate skill & Shared occ. & $s_{\mathrm{ne}}$ & $s_{\mathrm{hop}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        nb, hp = row["neighbor"], row["hop"]
        match = esc(nb["cv_skill"]) if nb["shared"] else r"\textit{none}"
        lines.append(
            f'{esc(row["job_skill"])} & {match} & '
            f'${len(nb["shared"])}/{len(nb["union"])}$ & '
            f'${nb["neighbor"]:.3f}$ & ${hp["hop"]:.3f}$ \\\\'
        )
    lines += [
        r"\midrule",
        r"\multicolumn{3}{@{}l}{Mean over required skills} & "
        f'$\\mathbf{{{totals["neighbor"]:.3f}}}$ & $\\mathbf{{{totals["hop"]:.3f}}}$ \\\\',
        r"\multicolumn{3}{@{}l}{Exact ESCO match ($T_{SY}^{esco}$)} & "
        f'\\multicolumn{{2}}{{c}}{{${totals["sy"]:.3f}$}} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def preamble(title: str) -> str:
    """Standalone LaTeX preamble shared by every panel (about 320pt wide, like example_esco.pdf)."""
    return rf"""% ============================================================================
%  {title} -- generated by pipeline/utils/kb_explanation.py -- do not edit by hand
%
%  One panel of the worked example for job {ID_J} x cv {ID_C} (see the module docstring).
%  All labels, counts and scores are read from the ESCO v1.2.1 graph.
%
%  Compile: pdflatex {title}        (needs TikZ, no extra fonts)
% ============================================================================
\documentclass[border=4pt]{{standalone}}
\usepackage{{tikz}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath}}
\renewcommand{{\familydefault}}{{\sfdefault}}
\usetikzlibrary{{arrows.meta,positioning,backgrounds,calc}}

% --- palette shared with example_esco.tex ----------------------------------
\definecolor{{occLeaf}}{{HTML}}{{{COLORS['occLeaf']}}}
\definecolor{{skLeaf}}{{HTML}}{{{COLORS['skLeaf']}}}
\definecolor{{skGrp}}{{HTML}}{{{COLORS['skGrp']}}}
\definecolor{{shared}}{{HTML}}{{{COLORS['shared']}}}
\definecolor{{faraway}}{{HTML}}{{{COLORS['far']}}}

\begin{{document}}
\begin{{tikzpicture}}[
  font=\footnotesize,
  node/.style={{draw=black,line width=0.7pt,rounded corners=2.5pt,align=center,
               text width=25mm,inner sep=2.2pt,minimum height=6.5mm}},
  skL/.style={{node,fill=skLeaf}},
  skG/.style={{node,fill=skGrp,text width=26mm}},
  occN/.style={{node,fill=occLeaf,text width=33mm,minimum height=5mm,font=\scriptsize}},
  occDead/.style={{occN,fill=occLeaf!35,draw=black!45,densely dashed}},
  bridge/.style={{draw=shared,line width=0.9pt}},
  dead/.style={{draw=black!35,line width=0.7pt,densely dotted}},
  broader/.style={{densely dashed,draw=black!55,line width=0.8pt}},
  near/.style={{draw=shared,line width=1.5pt}},
  detour/.style={{draw=faraway!45,line width=3.4pt,line cap=round,line join=round}},
  score/.style={{align=center,font=\footnotesize}},
  more/.style={{align=center,font=\scriptsize,text=black!60}},
  cross/.style={{font=\normalsize,text=faraway}},
  hdr/.style={{font=\small\itshape,anchor=center}},
]
"""


TAIL = "\n\\end{tikzpicture}\n\\end{document}\n"

# x of the three columns of an occupation panel: job skill, occupations, CV skill
COL_J, COL_O, COL_C = 0.0, 4.45, 8.9


def esc(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_")


def headers(top: float, middle: str) -> list[str]:
    """Column captions of an occupation panel."""
    return [
        rf"\node[hdr] at ({COL_J},{top:.2f}) {{job requires}};",
        rf"\node[hdr] at ({COL_O},{top:.2f}) {{{middle}}};",
        rf"\node[hdr] at ({COL_C},{top:.2f}) {{candidate has}};",
    ]


def bridge_panel(row: dict) -> str:
    """Panel of a job skill with shared occupations: up to three of them and the Jaccard score."""
    nb = row["neighbor"]
    shown = sorted(nb["shared"])[:3]
    span = (len(shown) - 1) * 0.85
    parts = headers(span / 2 + 1.05, "ESCO occupations requiring both")
    parts.append(rf"\node[skL] (jl) at ({COL_J},0) {{{esc(row['job_skill'])}}};")
    parts.append(rf"\node[skL] (cl) at ({COL_C},0) {{{esc(nb['cv_skill'])}}};")
    for k, occ in enumerate(shown):
        oy = span / 2 - k * 0.85
        parts.append(rf"\node[occN] (o{k}) at ({COL_O},{oy:.2f}) {{{esc(occ)}}};")
        parts.append(rf"\draw[bridge] (jl) -- (o{k});")
        parts.append(rf"\draw[bridge] (o{k}) -- (cl);")
    below = f"o{len(shown) - 1}"
    if len(nb["shared"]) > len(shown):
        parts.append(
            rf"\node[more,below=2.5mm of {below}] (more) "
            rf"{{$+\,{len(nb['shared']) - len(shown)}$ more}};"
        )
        below = "more"
    parts.append(
        rf"\node[score,below=3mm of {below}] "
        rf"{{$\dfrac{{|{len(nb['shared'])}|}}{{|{len(nb['union'])}|}} = {nb['neighbor']:.3f}$}};"
    )
    return "  " + "\n  ".join(parts)


def orphan_panel(row: dict) -> str:
    """Panel of a job skill that shares no occupation with the CV skills."""
    nb = row["neighbor"]
    parts = headers(1.05, "ESCO occupations requiring it")
    parts += [
        rf"\node[skL] (jorph) at ({COL_J},0) {{{esc(row['job_skill'])}}};",
        rf"\node[occDead] (odead) at ({COL_O},0) {{{esc(', '.join(sorted(nb['occ_job'])))}}};",
        r"\draw[dead] (jorph) -- (odead);",
        rf"\draw[dead] (odead) -- ({COL_O + 3.1},0);",
        rf"\node[cross] at ({COL_C},0) {{$\times$~\scriptsize no bridge}};",
        rf"\node[score,below=4mm of odead] "
        rf"{{$\dfrac{{|0|}}{{|{len(nb['union'])}|}} = 0.000$}};",
    ]
    return "  " + "\n  ".join(parts)


def taxonomy_panel(tax_paths: dict) -> str:
    """Panel of the skill taxonomy: the short path between the matched skills and the long one."""
    near_a, near_b = tax_paths["near"][0], tax_paths["near"][-1]
    far, left_chain, right_chain = tax_paths["far"], tax_paths["left_chain"], tax_paths["right_chain"]

    parts = [rf"\node[skG] (root) at (3.30,0) {{{esc(tax_paths['root'])}}};"]
    for k, node in enumerate(left_chain):
        parts.append(rf"\node[skG] (l{k}) at (1.15,{-1.35 * (k + 1):.2f}) {{{esc(node)}}};")
    for k, node in enumerate(right_chain):
        parts.append(rf"\node[skG] (r{k}) at (6.60,{-1.35 * (k + 1):.2f}) {{{esc(node)}}};")
    leaf_y = -1.35 * (len(left_chain) + 1)
    far_y = -1.35 * (len(right_chain) + 1)
    parts.append(rf"\node[skL] (na) at (0.00,{leaf_y:.2f}) {{{esc(near_a)}}};")
    parts.append(rf"\node[skL] (nb) at (2.45,{leaf_y:.2f}) {{{esc(near_b)}}};")
    parts.append(rf"\node[skL] (fa) at (6.60,{far_y:.2f}) {{{esc(far[0])}}};")
    parts.append(r"\draw[broader] (root) -- (l0);")
    parts.append(r"\draw[broader] (root) -- (r0);")
    for k in range(len(left_chain) - 1):
        parts.append(rf"\draw[broader] (l{k}) -- (l{k + 1});")
    for k in range(len(right_chain) - 1):
        parts.append(rf"\draw[broader] (r{k}) -- (r{k + 1});")
    last_l, last_r = len(left_chain) - 1, len(right_chain) - 1
    parts.append(rf"\draw[near] (l{last_l}) -- (na);")
    parts.append(rf"\draw[near] (l{last_l}) -- (nb);")
    parts.append(rf"\draw[broader] (r{last_r}) -- (fa);")
    evidence = tax_paths["near_evidence"]
    parts.append(
        rf"\node[score,anchor=north] at (1.22,{leaf_y - 0.62:.2f}) "
        rf"{{$d={evidence['distance']:.0f} \Rightarrow \varphi={evidence['hop']:.3f}$}};"
    )
    parts.append(
        rf"\node[score,anchor=north,text=black!55] at (6.60,{far_y - 0.62:.2f}) "
        rf"{{$d={len(far) - 1} > 6 \Rightarrow \varphi=0.000$}};"
    )
    chain = ["fa", *[f"r{k}" for k in reversed(range(len(right_chain)))],
             "root", *[f"l{k}" for k in range(len(left_chain))], "nb"]
    parts.append(r"\begin{scope}[on background layer]")
    parts.append(r"\draw[detour] " + " -- ".join(f"({n})" for n in chain) + ";")
    parts.append(rf"\node[more,text=black!55] at (3.87,-2.70) "
                 rf"{{{len(far) - 1} hops:\\the long way round}};")
    parts.append(r"\end{scope}")
    return "  " + "\n  ".join(parts)


def tikz_figures(rows: list[dict], tax_paths: dict) -> list[str]:
    """One standalone picture per matched job skill, one for the unmatched one, one for the taxonomy."""
    matched = [r for r in rows if r["neighbor"]["shared"]]
    orphan = next(r for r in rows if not r["neighbor"]["shared"])
    bodies = [bridge_panel(r) for r in matched]
    bodies.append(orphan_panel(orphan))
    bodies.append(taxonomy_panel(tax_paths))
    return [preamble(f"expl{i}.tex") + body + TAIL for i, body in enumerate(bodies, 1)]


def main() -> None:
    graph = KBGraph.load(ROOT / "cache/occSkill_graph.pkl")
    jobs = {j["id_j"]: j for j in json.loads((DATASET / "source/filtered_jobs_dt.json").read_text())}
    cvs = {c["id_c"]: c for c in json.loads((DATASET / "source/filtered_cvs_dt.json").read_text())}
    job, cv = jobs[ID_J], cvs[ID_C]

    rows = [explain_job_skill(graph, s, cv["esco_skills"]) for s in sorted(job["esco_skills"])]
    totals = {
        "neighbor": sum(r["neighbor"]["neighbor"] for r in rows) / len(rows),
        "hop": sum(r["hop"]["hop"] for r in rows) / len(rows),
    }

    # The figures must agree with the scores in the dataset.
    fitness = pd.read_csv(DATASET / "skill_fitness.csv")
    truth = fitness[(fitness.id_j == ID_J) & (fitness.id_c == ID_C)].iloc[0]
    totals["sy"] = float(truth.fit_skills_sy_esco)
    for arm, column in (("neighbor", "fit_skills_kb_neighbor_esco"), ("hop", "fit_skills_kb_hop_esco")):
        assert abs(totals[arm] - truth[column]) < 5e-4, (
            f"{arm}: recomputed {totals[arm]:.4f} != dataset {truth[column]:.4f}"
        )

    print(describe(rows, job, cv, totals))
    print()
    print(markdown_table(rows, totals))
    print()

    # Taxonomy panel: the first matched pair, and the path from the unmatched job skill to it.
    near = next(r["neighbor"] for r in rows if r["neighbor"]["shared"])
    orphan = next(r for r in rows if not r["neighbor"]["shared"])
    far_path = graph.path(
        graph.get_uri(orphan["job_skill"], reduce="Down"),
        graph.get_uri(near["cv_skill"], reduce="Down"),
        labeled=True, intermediate_properties={"type": "skill"},
    )
    # far_path goes from the unmatched skill up to the root and down to the matched skill.
    near_up = graph.path(
        graph.get_uri(near["cv_skill"], reduce="Down"),
        graph.get_uri(far_path[len(far_path) // 2], reduce="Down"),
        labeled=True, intermediate_properties={"type": "skill"},
    )
    root = next(n for n in far_path if n in near_up and n not in near["path"])
    split = far_path.index(root)
    right_chain = list(reversed(far_path[1:split]))  # from the root down to the unmatched skill
    left_chain = far_path[split + 1:-1]              # from the root down to the matched skill
    tax_paths = {
        "near": [near["job_skill"], near["cv_skill"]],
        "far": far_path,
        "root": root,
        "left_chain": left_chain,
        "right_chain": right_chain,
        "near_evidence": near,
    }

    LATEX.mkdir(exist_ok=True)
    (LATEX / "kb_explanation_table.tex").write_text(latex_table(rows, totals))
    written = []
    for i, body in enumerate(tikz_figures(rows, tax_paths), 1):
        tex = LATEX / f"expl{i}.tex"
        tex.write_text(body)
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=LATEX, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-3000:], file=sys.stderr)
            raise SystemExit(f"pdflatex failed on {tex.name}")
        for junk in ("aux", "log"):
            tex.with_suffix(f".{junk}").unlink(missing_ok=True)
        written.append(tex.stem)

    print(f"wrote {', '.join(f'kb_expl_examples/{n}.tex+.pdf' for n in written)}, "
          f"{(LATEX / 'kb_explanation_table.tex').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
