"""Local, deterministic resume-vs-JD scoring: keyword coverage via spaCy
noun-chunk/entity extraction. No LLM call -- fast and reproducible."""

from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# Generic head nouns whose modifiers ARE worth salvaging (e.g. "Kubernetes
# experience" -> "kubernetes"): these heads describe a skill/knowledge
# *claim*, so whatever qualifies them is usually the real skill/tool name.
_SKILL_CONTEXT_HEAD_NOUNS = {
    "experience", "year", "years", "knowledge", "understanding",
    "familiarity", "proficiency", "expertise", "background",
}

# Job-title / role head nouns whose modifiers describe the ROLE, not a
# skill claim (e.g. "Backend-focused Software Developer Intern") -- these
# are dropped whole, with no modifier salvage, so stylistic role framing
# doesn't get treated as a skill/tool keyword (and, in the truthfulness
# checker, doesn't get flagged as a fabricated claim).
_TITLE_HEAD_NOUNS = {
    "engineer", "developer", "professional", "intern", "manager",
    "specialist", "coordinator", "analyst", "scientist", "consultant",
}

# Other generic head nouns that are pure boilerplate either way (JD
# filler, not skill-context, not a title) -- dropped whole, no salvage.
_OTHER_GENERIC_HEAD_NOUNS = {
    "ability", "candidate", "candidates", "role", "team", "work",
    "environment", "skill", "skills", "plus", "etc", "responsibility",
    "responsibilities", "requirement", "requirements", "job", "position",
    "company", "bonus",
}

_GENERIC_HEAD_NOUNS = _SKILL_CONTEXT_HEAD_NOUNS | _TITLE_HEAD_NOUNS | _OTHER_GENERIC_HEAD_NOUNS

# Generic descriptive adjectives that shouldn't be kept even when they
# modify a real skill term (e.g. "strong" in "strong Python skills").
_GENERIC_ADJECTIVES = {
    "strong", "solid", "proven", "good", "excellent", "ideal", "hand",
    "hands", "extensive", "deep", "basic", "advanced", "great",
}

MIN_KEYWORD_LEN = 2
# Real skill/tool phrases are almost never more than a handful of words
# ("Large Language Model", "JWT-based authentication"). Poorly-punctuated
# source text (e.g. a PDF table flattened into one run-on line) can make
# spaCy's dependency parser chain many nouns into one giant noun chunk --
# cap word count so that parsing artifact never becomes a "keyword".
MAX_KEYWORD_WORDS = 5


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
        if not tokens or len(tokens) > MAX_KEYWORD_WORDS:
            return None
        term = " ".join(t.text.lower() for t in tokens)
        return term if len(term) >= MIN_KEYWORD_LEN else None

    if root_text not in _SKILL_CONTEXT_HEAD_NOUNS:
        # Title/role or pure-filler head (e.g. "Software Developer Intern",
        # "a bonus") -- drop the whole chunk, no modifier salvage.
        return None

    # Root is a skill-context head noun (e.g. "Kubernetes experience") --
    # salvage the compound/proper-noun modifiers, usually the real skill.
    modifiers = [
        t for t in span
        if t.i != root.i and t.is_alpha and not t.is_stop
        and t.text.lower() not in _GENERIC_ADJECTIVES
        and (t.dep_ == "compound" or t.pos_ == "PROPN")
    ]
    if not modifiers or len(modifiers) > MAX_KEYWORD_WORDS:
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
    """Crude, symmetric normalization used only for the substring match
    (never for display): lowercase, turn punctuation (hyphens, commas,
    parens, ...) into spaces so tokenization lines up with how
    extract_keywords builds its space-joined terms, and strip a single
    trailing 's' from words longer than 3 chars. Keeps things like
    'fine-tuning' vs 'fine tuning' and 'APIs' vs 'API' lining up without
    relying on spaCy's lemmatizer, which can mangle unfamiliar technical
    terms."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = []
    for word in normalized.split():
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
