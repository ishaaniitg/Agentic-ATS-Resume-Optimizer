"""Hybrid resume-vs-JD scoring: 50% semantic embedding similarity
(sentence-transformers) + 50% deterministic keyword coverage (spaCy
noun-chunk/entity extraction). No LLM call -- both halves are local
models, fast and reproducible (semantic similarity is not literally
deterministic-by-definition the way substring matching is, but the same
model + same inputs always produce the same score)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy

_NLP = None
_EMBEDDER = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


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

# Weights for the hybrid score.
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5

# Raw MiniLM cosine similarities for genuinely related professional text
# cluster in a fairly narrow band (rarely near 0 or 1 even for a great
# match) -- rescale that observed band to the full 0-100 range instead of
# using raw cosine directly, which would compress every score into the
# low-to-mid range regardless of how good the match actually is.
SEMANTIC_SIM_FLOOR = 0.15   # ~unrelated text
SEMANTIC_SIM_CEIL = 0.75    # ~strongly related text

# A missing keyword whose embedding is still this similar to the resume's
# best-matching line is probably present under different wording, not a
# real gap.
CONCEPTUAL_MATCH_THRESHOLD = 0.45


@dataclass
class ScoreResult:
    score: float  # 0-100, combined hybrid score
    keyword_score: float  # 0-100, exact/partial keyword coverage
    semantic_score: float  # 0-100, embedding cosine similarity (rescaled)
    matched_keywords: list[str]
    missing_keywords: list[str]  # exact lexical misses (includes conceptual_gaps)
    conceptual_gaps: list[str] = field(default_factory=list)  # subset of missing_keywords that also fail the semantic check -- likely genuine gaps, not just different phrasing

    @property
    def total_keywords(self) -> int:
        return len(self.matched_keywords) + len(self.missing_keywords)


def _clean_term(span) -> list[str]:
    """Turn a noun chunk / entity span into zero or more keyword strings.
    Uses raw token text (not spaCy lemmas) so unfamiliar technical proper
    nouns like "Kubernetes" aren't mangled by the small model's
    lemmatizer (which otherwise reduces it to "kubernete")."""
    root = span.root
    root_text = root.text.lower()

    if root.is_alpha and not root.is_stop and root_text not in _GENERIC_HEAD_NOUNS:
        tokens = [
            t for t in span
            if t.is_alpha and not t.is_stop and t.text.lower() not in _GENERIC_ADJECTIVES
        ]
        if not tokens or len(tokens) > MAX_KEYWORD_WORDS:
            return []
        term = " ".join(t.text.lower() for t in tokens)
        return [term] if len(term) >= MIN_KEYWORD_LEN else []

    if root_text not in _SKILL_CONTEXT_HEAD_NOUNS:
        # Title/role or pure-filler head (e.g. "Software Developer Intern",
        # "a bonus") -- drop the whole chunk, no modifier salvage.
        return []

    # Root is a skill-context head noun (e.g. "Kubernetes experience") --
    # salvage the compound/proper-noun modifiers, usually the real skill.
    modifiers = sorted(
        (
            t for t in span
            if t.i != root.i and t.is_alpha and not t.is_stop
            and t.text.lower() not in _GENERIC_ADJECTIVES
            and (t.dep_ == "compound" or t.pos_ == "PROPN")
        ),
        key=lambda t: t.i,
    )
    if not modifiers:
        return []

    # Group into contiguous runs so a conjunction like "Flask and Docker
    # experience" yields two separate keywords ("flask", "docker") instead
    # of one unmatchable glued phrase ("flask docker") -- "and" isn't a
    # modifier itself (filtered above as a stopword), so it splits the
    # token-index run.
    runs: list[list] = []
    for t in modifiers:
        if runs and t.i == runs[-1][-1].i + 1:
            runs[-1].append(t)
        else:
            runs.append([t])

    terms = []
    for run in runs:
        if len(run) > MAX_KEYWORD_WORDS:
            continue
        term = " ".join(tok.text.lower() for tok in run)
        if len(term) >= MIN_KEYWORD_LEN and term not in _GENERIC_HEAD_NOUNS:
            terms.append(term)
    return terms


def extract_keywords(text: str) -> list[str]:
    """Pull candidate skill/requirement keywords from text via spaCy noun
    chunks and named entities, deduplicated."""
    doc = _get_nlp()(text)
    keywords: set[str] = set()

    for chunk in doc.noun_chunks:
        keywords.update(_clean_term(chunk))

    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "PERSON", "DATE", "CARDINAL"}:
            continue  # not skill-relevant
        keywords.update(_clean_term(ent))

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


def _keyword_coverage(resume_text: str, jd_text: str) -> tuple[float, list[str], list[str]]:
    """Percent of JD keywords found (via folded substring match) in the
    resume text. Returns (score, matched, missing)."""
    jd_keywords = extract_keywords(jd_text)
    if not jd_keywords:
        return 0.0, [], []

    resume_folded = _fold(resume_text)
    matched, missing = [], []
    for kw in jd_keywords:
        (matched if _fold(kw) in resume_folded else missing).append(kw)

    score = round(100 * len(matched) / len(jd_keywords), 2)
    return score, matched, missing


def _cosine_similarity(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _resume_lines(resume_text: str) -> list[str]:
    return [line.strip() for line in resume_text.splitlines() if line.strip()]


def _jd_sentences(jd_text: str) -> list[str]:
    doc = _get_nlp()(jd_text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _semantic_score(resume_text: str, jd_text: str) -> float:
    """For each JD sentence, find its best-matching resume line (cosine
    similarity), then average across JD sentences. This max-pooling
    alignment -- not one whole-document-vs-whole-document embedding --
    matters because a resume has plenty of content (contact info, dates,
    education tables) that has nothing to do with the JD; averaging that
    in dilutes a perfectly good match down to a mediocre score."""
    resume_lines = _resume_lines(resume_text)
    jd_sentences = _jd_sentences(jd_text)
    if not resume_lines or not jd_sentences:
        return 0.0

    embedder = _get_embedder()
    resume_vecs = embedder.encode(resume_lines)
    jd_vecs = embedder.encode(jd_sentences)

    best_sims = [
        max(_cosine_similarity(jd_vec, r_vec) for r_vec in resume_vecs)
        for jd_vec in jd_vecs
    ]
    raw = sum(best_sims) / len(best_sims)
    rescaled = (raw - SEMANTIC_SIM_FLOOR) / (SEMANTIC_SIM_CEIL - SEMANTIC_SIM_FLOOR)
    return round(100 * max(0.0, min(1.0, rescaled)), 2)


def _find_conceptual_gaps(missing_keywords: list[str], resume_text: str) -> list[str]:
    """Of the exact-match misses, find the ones that are ALSO not
    semantically present anywhere in the resume -- i.e. likely genuine
    skill gaps rather than the same skill phrased differently."""
    if not missing_keywords:
        return []

    resume_lines = _resume_lines(resume_text)
    if not resume_lines:
        return list(missing_keywords)

    embedder = _get_embedder()
    line_vecs = embedder.encode(resume_lines)
    keyword_vecs = embedder.encode(missing_keywords)

    gaps = []
    for keyword, kw_vec in zip(missing_keywords, keyword_vecs):
        best_sim = max(_cosine_similarity(kw_vec, line_vec) for line_vec in line_vecs)
        if best_sim < CONCEPTUAL_MATCH_THRESHOLD:
            gaps.append(keyword)
    return gaps


def score_resume(resume_text: str, jd_text: str) -> ScoreResult:
    """Hybrid ATS score: 50% semantic (embedding cosine similarity between
    the whole resume and JD) + 50% keyword coverage (exact/partial match
    of extracted JD skill keywords against the resume text)."""
    keyword_score, matched, missing = _keyword_coverage(resume_text, jd_text)
    semantic_score = _semantic_score(resume_text, jd_text)
    combined = round(SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score, 2)
    conceptual_gaps = _find_conceptual_gaps(missing, resume_text)

    return ScoreResult(
        score=combined,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        matched_keywords=matched,
        missing_keywords=missing,
        conceptual_gaps=conceptual_gaps,
    )
