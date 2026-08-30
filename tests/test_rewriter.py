"""Quick Phase 3 test: truthfulness check on synthetic before/after text
pairs. No LLM call needed for this part -- it only exercises the
deterministic verification step."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rewriter import check_truthfulness

ORIGINAL_RESUME = """
Software Developer Intern at Samsung R&D. Built REST APIs using Flask and
Python, deployed with Docker on AWS SageMaker for Large Language Model
fine-tuning. Built a food ordering platform using Node.js, Express.js,
MongoDB and React.
"""


def test_faithful_rephrase_passes():
    # Same facts, just reworded/reordered -- should pass.
    rewrite = """
    Backend-focused Software Developer Intern who built and deployed REST
    APIs (Flask, Python) via Docker on AWS SageMaker to support Large
    Language Model fine-tuning. Also engineered a food ordering platform
    with React, Node.js, Express.js, and MongoDB.
    """
    result = check_truthfulness(rewrite, ORIGINAL_RESUME)
    print(f"[INFO] faithful rewrite unverified terms: {result.unverified_terms}")
    assert result.passed, f"expected a faithful rephrase to pass, got unverified: {result.unverified_terms}"
    print("[PASS] faithful rephrase passes truthfulness check")


def test_fabricated_skill_is_rejected():
    # Injects "Kubernetes" and "GraphQL", neither in the original -- must fail.
    rewrite = """
    Software Developer Intern who deployed containerized services with
    Docker and Kubernetes, and built GraphQL APIs using Flask and Python.
    """
    result = check_truthfulness(rewrite, ORIGINAL_RESUME)
    print(f"[INFO] fabricated rewrite unverified terms: {result.unverified_terms}")
    assert not result.passed, "expected fabricated skills to fail the truthfulness check"
    assert any("kubernetes" in t for t in result.unverified_terms)
    assert any("graphql" in t for t in result.unverified_terms)
    print("[PASS] fabricated skills (Kubernetes, GraphQL) correctly rejected")


if __name__ == "__main__":
    test_faithful_rephrase_passes()
    test_fabricated_skill_is_rejected()
    print("\nPhase 3 rewriter (truthfulness check) tests completed.")
