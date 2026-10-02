"""System prompt and job/CV text blocks of the LLM relevance judge (used by 2_1_llm_relevance)."""

SYSTEM_PROMPT = (
    "You are an expert technical recruiter deciding whether a candidate should be suitable for a job offer.\n"
    "You will receive a JOB OFFER (occupation, years of experience required, education level, required skills, required languages, description) and a CANDIDATE CV "
    "(years of experience, education level, languages, skills).\n"
    "IMPORTANT: do NOT use ESCO (European Skills, Competences, Qualifications and Occupations) nor O*NET (Occupational Information Network). "
    "Do not map occupations or skills to ESCO or O*NET codes, concepts, groups or hierarchies, and do not use ESCO or O*NET skill-occupation relations to decide the match. "
    "Judge only from the text of the job offer and the CV, as a human recruiter would.\n"
    "Evaluate the match by considering, in order of importance:\n"
    "1. SKILL FIT: compare skills by meaning, not wording. Two skills match if a recruiter would consider them the same competence or close substitutes (e.g., "
    "different tools or terms within the same specialty). Do not reward word overlap between unrelated competencies, and do not penalize different wording for the "
    "same competence.\n"
    "2. EXPERIENCE, EDUCATION, LANGUAGES: they can lower the score when clearly insufficient for the job's requirements. "
    "Education is on the EQF scale: 1-4 basic/secondary, 5 short-cycle higher, 6 Bachelor's, 7 Master's, 8 Doctorate.\n\n"
    "Assign exactly one score:\n"
    "0 = Bad fit : the candidate's field does not fit the job's occupation; a recruiter would discard this CV immediately.\n"
    "1 = Right or adjacent fit, but with clear gaps in required skills, experience, or education.\n"
    "2 = Fit: the field matches and the requirements are essentially met, with at most minor gap.\n\n"
    "Respond with a single digit: 0, 1, or 2. Nothing else."
)

# (label shown in the prompt, column in the dataframe); the list order is the line order.
JOB_FIELDS = [
    ("Occupation",                   "raw_occupation"),
    ("Years of experience required", "years_experience"),
    ("Education (EQF level)",        "edu_eqf"),
    ("Required skills",              "raw_skills"),
    ("Required languages",           "languages"),
    ("Job description",              "eng_description"),
]
CV_FIELDS = [
    ("Years of experience",   "years_experience"),
    ("Education (EQF level)", "edu_eqf"),
    ("Languages",             "languages"),
    ("Skills",                "raw_skills"),
]


def fmt(value) -> str:
    """Flatten a field value into a readable string; empty -> 'Not specified'.

    ['Java', 'SQL'] -> 'Java, SQL' ;  [] -> 'Not specified' ;  7 -> '7'
    """
    if isinstance(value, (list, tuple, set)):
        joined = ", ".join(str(x) for x in value if str(x).strip())
        return joined or "Not specified"
    return str(value).strip() or "Not specified"


def build_block(record, fields) -> str:
    """One 'Label: value' line per field, in the given order."""
    return "\n".join(f"{label}: {fmt(record[col])}" for label, col in fields)


if __name__ == "__main__":
    assert fmt(["Java", "SQL"]) == "Java, SQL"
    assert fmt([]) == "Not specified" and fmt("  ") == "Not specified"
    assert fmt(7) == "7"
    assert build_block({"a": ["x", "y"], "b": ""}, [("A", "a"), ("B", "b")]) == "A: x, y\nB: Not specified"
    print("ok")
