"""Quick Phase 3 test: truthfulness check on synthetic before/after text
pairs (no LLM call needed), plus key-rotation/model-fallback resilience
tests using a mocked Gemini client (no real API calls, no quota spent)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.rewriter as rewriter
from google.genai import errors as genai_errors
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


def _quota_error() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}}
    )


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    """Scripted fake for client.models -- `script` is a list of either
    'fail' or 'ok', one per call, in order."""

    def __init__(self, script):
        self._script = iter(script)
        self.calls = []  # records the model name used on each call

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = next(self._script)
        if outcome == "fail":
            raise _quota_error()
        return _FakeResponse("rewritten resume text")


class _FakeClient:
    def __init__(self, models: _FakeModels):
        self.models = models


def _reset_rewriter_globals():
    rewriter._ROTATOR = None
    rewriter._CLIENTS = {}


def test_key_rotation_skips_exhausted_key_and_succeeds():
    fake_models = _FakeModels(["fail", "ok"])  # key1 exhausted, key2 works
    fake_client = _FakeClient(fake_models)

    _reset_rewriter_globals()
    rewriter._ROTATOR = rewriter._KeyModelRotator(["key1", "key2"], ["model-a"])
    original_get_client = rewriter._get_client
    rewriter._get_client = lambda api_key: fake_client
    try:
        result = rewriter.generate_rewrite("original resume", "jd text")
    finally:
        rewriter._get_client = original_get_client
        _reset_rewriter_globals()

    print(f"[INFO] calls made: {fake_models.calls}")
    assert result == "rewritten resume text"
    assert fake_models.calls == ["model-a", "model-a"], "should retry immediately on the next key, same model"
    print("[PASS] key rotation skips an exhausted key and succeeds on the next, without sleeping")


def test_falls_back_to_secondary_model_after_all_keys_exhausted():
    # 2 keys x 2 models -- both keys fail on model-a, then model-b succeeds.
    fake_models = _FakeModels(["fail", "fail", "ok"])
    fake_client = _FakeClient(fake_models)

    _reset_rewriter_globals()
    rewriter._ROTATOR = rewriter._KeyModelRotator(["key1", "key2"], ["model-a", "model-b"])
    original_get_client = rewriter._get_client
    rewriter._get_client = lambda api_key: fake_client
    try:
        result = rewriter.generate_rewrite("original resume", "jd text")
        status = rewriter.get_active_model_status()
    finally:
        rewriter._get_client = original_get_client
        _reset_rewriter_globals()

    print(f"[INFO] calls made: {fake_models.calls}")
    print(f"[INFO] status after fallback: {status}")
    assert result == "rewritten resume text"
    assert fake_models.calls == ["model-a", "model-a", "model-b"]
    assert status["active_model"] == "model-b"
    print("[PASS] falls back to the secondary model once every key is exhausted on the primary")


def test_all_pairs_exhausted_raises_original_error_not_internal_wrapper():
    """When every (key, model) pair -- and the last-resort rate-limit
    retry -- is exhausted, callers (e.g. app.py's `except
    genai_errors.ClientError`) must still see a real google.genai error,
    not rewriter's internal retry-bookkeeping exception types."""
    fake_models = _FakeModels(["fail"] * 20)  # always fails
    fake_client = _FakeClient(fake_models)

    _reset_rewriter_globals()
    rewriter._ROTATOR = rewriter._KeyModelRotator(["key1"], ["model-a"])
    original_get_client = rewriter._get_client
    original_sleep = time.sleep
    rewriter._get_client = lambda api_key: fake_client
    time.sleep = lambda seconds: None  # skip the real 10-15s rate-limit backoff for this test
    try:
        try:
            rewriter.generate_rewrite("original resume", "jd text")
            raise AssertionError("expected a ClientError to be raised")
        except genai_errors.ClientError as exc:
            print(f"[INFO] correctly raised a real ClientError: {exc.code} {exc.status}")
            assert exc.code == 429
    finally:
        rewriter._get_client = original_get_client
        time.sleep = original_sleep
        _reset_rewriter_globals()

    print("[PASS] exhausting all retries raises the original ClientError, not an internal wrapper type")


if __name__ == "__main__":
    test_faithful_rephrase_passes()
    test_fabricated_skill_is_rejected()
    test_key_rotation_skips_exhausted_key_and_succeeds()
    test_falls_back_to_secondary_model_after_all_keys_exhausted()
    test_all_pairs_exhausted_raises_original_error_not_internal_wrapper()
    print("\nPhase 3 rewriter tests completed.")
