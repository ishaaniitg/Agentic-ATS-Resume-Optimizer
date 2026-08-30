"""Phase 3 orchestration: iterate Generator -> truthfulness check -> Judge
(Phase 2's deterministic scorer) until target_score is hit or
max_iterations is exhausted. Keeps the best-scoring version seen across
all iterations, not just the last one, and logs a before/after per section
for every accepted rewrite."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rewriter import check_truthfulness, generate_rewrite
from src.scorer import ScoreResult, score_resume

DEFAULT_TARGET_SCORE = 80.0
DEFAULT_MAX_ITERATIONS = 5
MAX_TRUTHFULNESS_RETRIES = 2

_SECTION_HEADERS = {
    "summary", "objective", "skills", "experience", "education",
    "projects", "certifications",
}


def split_sections(text: str) -> dict[str, str]:
    """Best-effort split of resume text into sections by common headers,
    for before/after logging (not for scoring or generation)."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        stripped = line.strip().lower().rstrip(":")
        if stripped in _SECTION_HEADERS:
            current = stripped
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def diff_sections(before: str, after: str) -> dict[str, tuple[str, str]]:
    """Return {section_name: (before_text, after_text)} for sections whose
    content changed between two resume drafts."""
    before_sections = split_sections(before)
    after_sections = split_sections(after)
    changed = {}
    for name in before_sections.keys() | after_sections.keys():
        b, a = before_sections.get(name, ""), after_sections.get(name, "")
        if b != a:
            changed[name] = (b, a)
    return changed


@dataclass
class IterationLog:
    iteration: int
    score: float
    accepted: bool
    rejection_reason: str | None = None
    changed_sections: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class LoopResult:
    best_resume_text: str
    best_score: float
    iterations_run: int
    hit_target: bool
    history: list[IterationLog] = field(default_factory=list)


def _feedback_from_score(result: ScoreResult) -> str:
    if not result.missing_keywords:
        return (
            "All detected JD keywords are already covered; focus on stronger "
            "phrasing and quantifiable impact in existing bullet points."
        )
    missing = ", ".join(result.missing_keywords[:10])
    return (
        f"The resume is still missing these JD-relevant terms: {missing}. "
        "Only surface ones you can truthfully draw from the original resume "
        "-- do not fabricate experience with anything not already present."
    )


def run_rewrite_loop(
    original_resume_text: str,
    jd_text: str,
    target_score: float = DEFAULT_TARGET_SCORE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> LoopResult:
    """Generator -> truthfulness check -> Judge (Phase 2 scorer) loop.
    Returns the best-scoring resume seen, even if a later iteration
    regressed or was rejected for truthfulness."""
    initial_score = score_resume(original_resume_text, jd_text)
    history = [IterationLog(iteration=0, score=initial_score.score, accepted=True)]

    best_text, best_score = original_resume_text, initial_score.score
    if best_score >= target_score:
        return LoopResult(best_text, best_score, 0, True, history)

    feedback = _feedback_from_score(initial_score)
    current_text = original_resume_text

    for i in range(1, max_iterations + 1):
        candidate = None
        for _attempt in range(1, MAX_TRUTHFULNESS_RETRIES + 1):
            draft = generate_rewrite(current_text, jd_text, feedback=feedback)
            truthfulness = check_truthfulness(draft, original_resume_text)
            if truthfulness.passed:
                candidate = draft
                break
            feedback = (
                f"{feedback}\n\nYour previous attempt introduced unverified claims "
                f"not present in the original resume: "
                f"{', '.join(truthfulness.unverified_terms)}. Remove or rephrase "
                "those -- do not invent skills."
            )

        if candidate is None:
            history.append(IterationLog(
                i, best_score, accepted=False,
                rejection_reason="failed truthfulness check after retries",
            ))
            continue

        candidate_score = score_resume(candidate, jd_text)
        history.append(IterationLog(
            i, candidate_score.score, accepted=True,
            changed_sections=diff_sections(current_text, candidate),
        ))

        current_text = candidate
        if candidate_score.score > best_score:
            best_score, best_text = candidate_score.score, candidate

        if best_score >= target_score:
            return LoopResult(best_text, best_score, i, True, history)

        feedback = _feedback_from_score(candidate_score)

    return LoopResult(best_text, best_score, max_iterations, False, history)
