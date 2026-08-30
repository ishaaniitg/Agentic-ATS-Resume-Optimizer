"""Quick Phase 2 test: verifies src.scorer.score_resume produces sane
keyword coverage on a synthetic resume/JD pair, and sanity-checks it
against a real parsed resume (Phase 1 + Phase 2 combined)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scorer import score_resume

JD_TEXT = """
We are hiring a Software Engineer with strong experience in Python and
Docker. The ideal candidate has hands-on experience with Flask, AWS, and
Large Language Models. Familiarity with React and MongoDB is a bonus.
Kubernetes experience is required.
"""

GOOD_RESUME = """
Software Developer Intern. Built REST APIs using Flask and Python, deployed
with Docker on AWS SageMaker for Large Language Model fine-tuning. Built a
dynamic platform using React and MongoDB.
"""

WEAK_RESUME = """
Marketing coordinator with experience in social media campaigns, content
writing, and customer outreach.
"""


def test_good_resume_scores_high_and_flags_missing():
    result = score_resume(GOOD_RESUME, JD_TEXT)
    print(f"[INFO] good resume score: {result.score}")
    print(f"[INFO] matched: {result.matched_keywords}")
    print(f"[INFO] missing: {result.missing_keywords}")
    assert result.score > 50, "expected a strongly matching resume to score above 50"
    assert any("kubernetes" in kw for kw in result.missing_keywords), (
        "Kubernetes is not mentioned in the resume and should show as missing"
    )
    print("[PASS] good resume scores high, correctly flags missing Kubernetes")


def test_weak_resume_scores_low():
    result = score_resume(WEAK_RESUME, JD_TEXT)
    print(f"[INFO] weak resume score: {result.score}")
    assert result.score < 30, "expected an unrelated resume to score low"
    print("[PASS] unrelated resume scores low")


def test_empty_jd_returns_zero_without_crashing():
    result = score_resume(GOOD_RESUME, "")
    assert result.score == 0.0
    assert result.matched_keywords == [] and result.missing_keywords == []
    print("[PASS] empty JD handled gracefully")


def test_real_cv_against_matching_jd():
    """Sanity check using the real parsed resume from Phase 1 (cv.pdf),
    scored against a JD written to match its actual content."""
    cv_path = os.path.join(os.path.dirname(__file__), "..", "cv.pdf")
    if not os.path.isfile(cv_path):
        print("[SKIP] cv.pdf not present, skipping real-resume sanity check")
        return

    from src.parser import parse_resume

    resume_text = parse_resume(cv_path)
    matching_jd = """
    Looking for a Software Developer Intern with experience in Flask,
    Docker, AWS SageMaker, Large Language Models, Node.js, Express.js,
    MongoDB, React.js, and JWT authentication.
    """
    result = score_resume(resume_text, matching_jd)
    print(f"[INFO] real cv.pdf score against tailored JD: {result.score}")
    print(f"[INFO] matched: {result.matched_keywords}")
    print(f"[INFO] missing: {result.missing_keywords}")
    assert result.score > 40, "expected the real resume to match its own tailored JD well"
    print("[PASS] real cv.pdf scores reasonably against a matching JD")


if __name__ == "__main__":
    test_good_resume_scores_high_and_flags_missing()
    test_weak_resume_scores_low()
    test_empty_jd_returns_zero_without_crashing()
    test_real_cv_against_matching_jd()
    print("\nPhase 2 scorer tests completed.")
