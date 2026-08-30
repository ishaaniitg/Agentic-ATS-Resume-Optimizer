"""Phase 4: candidate-facing and employer-facing entry points wrapping the
Phase 1-3 pipeline. Candidate mode scores and rewrites a single resume;
employer mode ranks many resumes against one job description using only
the fast, local Phase 1-2 pipeline (no LLM calls, no rewriting)."""

from __future__ import annotations

from dataclasses import dataclass

from src.agent_loop import DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_SCORE, run_rewrite_loop
from src.parser import ParseError, parse_resume
from src.scorer import score_resume


@dataclass
class CandidateResult:
    resume_path: str
    initial_score: float
    final_score: float
    final_resume_text: str
    hit_target: bool
    iterations_run: int


def candidate_mode(
    resume_path: str,
    jd_text: str,
    target_score: float = DEFAULT_TARGET_SCORE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> CandidateResult:
    """Single resume -> Phase 1-3 pipeline -> final resume + score."""
    resume_text = parse_resume(resume_path)
    initial_score = score_resume(resume_text, jd_text).score
    loop_result = run_rewrite_loop(resume_text, jd_text, target_score, max_iterations)

    return CandidateResult(
        resume_path=resume_path,
        initial_score=initial_score,
        final_score=loop_result.best_score,
        final_resume_text=loop_result.best_resume_text,
        hit_target=loop_result.hit_target,
        iterations_run=loop_result.iterations_run,
    )


@dataclass
class CandidateRanking:
    resume_path: str
    score: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    error: str | None = None


def employer_mode(resume_paths: list[str], jd_text: str) -> list[CandidateRanking]:
    """N resumes + one JD -> Phase 1-2 on each -> ranked list (highest
    score first) with each candidate's matched/missing gap summary. A
    resume that fails to parse is included with an error note rather than
    crashing the whole batch."""
    rankings: list[CandidateRanking] = []

    for path in resume_paths:
        try:
            resume_text = parse_resume(path)
            result = score_resume(resume_text, jd_text)
            rankings.append(CandidateRanking(
                resume_path=path,
                score=result.score,
                matched_keywords=result.matched_keywords,
                missing_keywords=result.missing_keywords,
            ))
        except (ParseError, ValueError) as exc:
            rankings.append(CandidateRanking(
                resume_path=path, score=0.0, matched_keywords=[],
                missing_keywords=[], error=str(exc),
            ))

    rankings.sort(key=lambda r: r.score, reverse=True)
    return rankings
