"""Live Phase 3 end-to-end check: real Gemini API calls against the real
cv.pdf. Not part of the deterministic test suite (costs real API calls) --
run manually to sanity-check the whole Generator -> truthfulness ->
Judge loop on real content."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent_loop import run_rewrite_loop
from src.parser import parse_resume
from src.rewriter import check_truthfulness
from src.scorer import score_resume

JD_TEXT = """
We're hiring a Backend/ML Software Engineer Intern. Ideal candidates have
experience with Flask, Docker, AWS, and fine-tuning Large Language Models.
Experience with Kubernetes, CI/CD pipelines, and GraphQL is a strong plus.
Also looking for familiarity with React, Node.js, MongoDB, and JWT-based
authentication from personal or academic projects.
"""


def main():
    cv_path = os.path.join(os.path.dirname(__file__), "..", "cv.pdf")
    resume_text = parse_resume(cv_path)

    initial = score_resume(resume_text, JD_TEXT)
    print(f"Initial score: {initial.score}")
    print(f"Initial missing keywords: {initial.missing_keywords}\n")

    result = run_rewrite_loop(resume_text, JD_TEXT, target_score=85.0, max_iterations=2)

    print("\n=== LOOP RESULT ===")
    print(f"iterations_run: {result.iterations_run}")
    print(f"hit_target: {result.hit_target}")
    print(f"best_score: {result.best_score} (started at {initial.score})")

    for log in result.history:
        print(
            f"  iter {log.iteration}: score={log.score} accepted={log.accepted}"
            f"{' reason=' + log.rejection_reason if log.rejection_reason else ''}"
            f" changed_sections={list(log.changed_sections.keys())}"
        )

    assert result.best_score >= initial.score, "loop should never end up worse than the original"

    final_truthfulness = check_truthfulness(result.best_resume_text, resume_text)
    print(f"\nFinal truthfulness check passed: {final_truthfulness.passed}")
    assert final_truthfulness.passed, f"final resume has unverified terms: {final_truthfulness.unverified_terms}"

    print("\n=== BEST RESUME (first 1200 chars) ===")
    print(result.best_resume_text[:1200])

    if result.iterations_run >= 1 and result.history[1].accepted:
        print("\n=== ITERATION 1 CHANGED SECTIONS ===")
        for section, (before, after) in result.history[1].changed_sections.items():
            print(f"\n--- {section} (before) ---\n{before[:300]}")
            print(f"--- {section} (after) ---\n{after[:300]}")

    print("\n[PASS] live Phase 3 end-to-end run completed successfully")


if __name__ == "__main__":
    main()
