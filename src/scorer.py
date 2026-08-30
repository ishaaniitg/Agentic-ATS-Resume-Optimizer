"""Local, deterministic resume-vs-JD scoring: keyword coverage via spaCy
noun-chunk/entity extraction. No LLM call -- fast and reproducible."""

from __future__ import annotations

from dataclasses import dataclass

import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# Generic head nouns that show up in JD boilerplate noun chunks (e.g.
# "strong experience", "ideal candidate", "Kubernetes experience") but
# aren't themselves meaningful skill/requirement keywords.
_GENERIC_HEAD_NOUNS = {
    "experience", "year", "years", "ability", "knowledge", "understanding",
    "candidate", "candidates", "role", "team", "work", "environment",
    "skill", "skills", "plus", "etc", "responsibility", "responsibilities",
    "requirement", "requirements", "job", "position", "company",
    "engineer", "developer", "professional", "familiarity", "bonus",
    "background", "proficiency", "expertise",
}

# Generic descriptive adjectives that shouldn't be kept even when they
# modify a real skill term (e.g. "strong" in "strong Python skills").
_GENERIC_ADJECTIVES = {
    "strong", "solid", "proven", "good", "excellent", "ideal", "hand",
    "hands", "extensive", "deep", "basic", "advanced", "great",
}

MIN_KEYWORD_LEN = 2


@dataclass
class ScoreResult:
    score: float  # 0-100, percent of JD keywords found in the resume
    matched_keywords: list[str]
    missing_keywords: list[str]

    @property
    def total_keywords(self) -> int:
        return len(self.matched_keywords) + len(self.missing_keywords)


def _clean_term(span) -> str | None:
    """Turn a noun chunk / entity span into a keyword string, or None if
    it's boilerplate. Uses raw token text (not spaCy lemmas) so unfamiliar
    technical proper nouns like "Kubernetes" aren't mangled by the small
    model's lemmatizer (which otherwise reduces it to "kubernete")."""
    root = span.root
    root_text = root.text.lower()

    if root.is_alpha and not root.is_stop and root_text not in _GENERIC_HEAD_NOUNS:
        tokens = [
            t for t in span
            if t.is_alpha and not t.is_stop and t.text.lower() not in _GENERIC_ADJECTIVES
        ]
        if not tokens:
            return None
        term = " ".join(t.text.lower() for t in tokens)
        return term if len(term) >= MIN_KEYWORD_LEN else None

    # Root is a generic head noun (e.g. "Kubernetes experience") -- salvage
    # the compound/proper-noun modifiers, which are usually the real skill.
    modifiers = [
        t for t in span
        if t.i != root.i and t.is_alpha and not t.is_stop
        and t.text.lower() not in _GENERIC_ADJECTIVES
        and (t.dep_ == "compound" or t.pos_ == "PROPN")
    ]
    if not modifiers:
        return None
    term = " ".join(t.text.lower() for t in modifiers)
    if len(term) < MIN_KEYWORD_LEN or term in _GENERIC_HEAD_NOUNS:
        return None
    return term


def extract_keywords(text: str) -> list[str]:
    """Pull candidate skill/requirement keywords from text via spaCy noun
    chunks and named entities, deduplicated."""
    doc = _get_nlp()(text)
    keywords: set[str] = set()

    for chunk in doc.noun_chunks:
        term = _clean_term(chunk)
        if term:
            keywords.add(term)

    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "PERSON", "DATE", "CARDINAL"}:
            continue  # not skill-relevant
        term = _clean_term(ent)
        if term:
            keywords.add(term)

    return sorted(keywords)


def _fold(text: str) -> str:
    """Crude, symmetric plural folding used only for the substring match
    (never for display): lowercase, strip a single trailing 's' from words
    longer than 3 chars. Keeps simple plural/singular mismatches (e.g.
    'APIs' vs 'API') lining up without relying on spaCy's lemmatizer, which
    can mangle unfamiliar technical terms."""
    words = []
    for word in text.lower().split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def score_resume(resume_text: str, jd_text: str) -> ScoreResult:
    """Score a resume against a job description via deterministic keyword
    coverage: percent of JD keywords found (via folded substring match) in
    the resume text."""
    jd_keywords = extract_keywords(jd_text)
    if not jd_keywords:
        return ScoreResult(score=0.0, matched_keywords=[], missing_keywords=[])

    resume_folded = _fold(resume_text)

    matched, missing = [], []
    for kw in jd_keywords:
        (matched if _fold(kw) in resume_folded else missing).append(kw)

    score = round(100 * len(matched) / len(jd_keywords), 2)
    return ScoreResult(score=score, matched_keywords=matched, missing_keywords=missing)
