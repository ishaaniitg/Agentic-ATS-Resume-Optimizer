"""LLM-backed resume rewriting (Gemini) plus a deterministic truthfulness
check that rejects any rewrite introducing skills/tech not present in the
originally parsed resume.

Resilience strategy against 429 RESOURCE_EXHAUSTED (Gemini's free-tier
quota is scoped per (project, model) -- a key exhausted on one model may
still work on another, and a fresh key may still work on the exhausted
model):
  1. On a quota error, rotate to the next (API key, model) pair that
     hasn't hit quota yet and retry immediately -- no need to wait, since
     it's a fresh quota bucket.
  2. Once every (key, model) combination has hit quota, fall back to a
     tenacity-managed wait-and-retry as a last resort.
  3. A transient error (server overload, network blip) is retried on the
     SAME (key, model) with backoff -- it isn't a quota problem, so
     rotating wouldn't help.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
import tenacity
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from src.scorer import extract_keywords, _fold

load_dotenv()

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
MODEL_CHAIN = [m for m in (PRIMARY_MODEL, FALLBACK_MODEL) if m]

# Last-resort wait window (seconds) once every (key, model) pair is
# exhausted -- only reached after key rotation and model fallback have
# both been tried.
RATE_LIMIT_WAIT_MIN_SECONDS = 10
RATE_LIMIT_WAIT_MAX_SECONDS = 15
MAX_RATE_LIMIT_ATTEMPTS = 3

MAX_TRANSIENT_RETRIES = 3
TRANSIENT_BACKOFF_SECONDS = 5

GENERATOR_SYSTEM_PROMPT = """You are a resume-tailoring assistant. Rewrite the
candidate's resume so it better matches the target job description.

Hard rules:
- Do NOT invent skills, tools, employers, job titles, degrees, dates, or
  achievements that are not already present in the original resume.
- You may rephrase, reorder, emphasize, and quantify existing bullet points,
  and surface existing skills more prominently, but every technology,
  skill, or claim in your output must be traceable to the original resume.
- Keep the same overall structure and section order as the original resume.
- Output ONLY the rewritten resume text -- no commentary, no markdown fences.
"""


def _load_api_keys() -> list[str]:
    multi = os.environ.get("GEMINI_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    if single:
        return [single]
    raise RuntimeError(
        "No Gemini API key configured -- set GEMINI_API_KEY or GEMINI_API_KEYS in .env"
    )


class _KeyModelRotator:
    """Round-robins across (API key, model) pairs, skipping any pair that
    has already hit a quota error this session. Exhausts every key on the
    primary model before falling back to the next model in the chain."""

    def __init__(self, keys: list[str], models: list[str]):
        self._keys = keys
        self._models = models
        self._pairs = [(k, m) for m in models for k in keys]
        self._idx = 0
        self._exhausted: set[tuple[str, str]] = set()

    def current(self) -> tuple[str, str]:
        return self._pairs[self._idx % len(self._pairs)]

    def mark_exhausted(self, key: str, model: str) -> None:
        self._exhausted.add((key, model))

    def advance(self) -> tuple[str, str] | None:
        """Move to the next non-exhausted pair, or None if all are spent."""
        for _ in range(len(self._pairs)):
            self._idx += 1
            candidate = self._pairs[self._idx % len(self._pairs)]
            if candidate not in self._exhausted:
                return candidate
        return None

    def reset_exhaustion(self) -> None:
        """Quota errors are daily, not permanent -- called after the
        last-resort wait succeeds, so a future run doesn't stay stuck
        skipping pairs that may have recovered."""
        self._exhausted.clear()

    def status(self) -> dict:
        key, model = self.current()
        return {
            "active_model": model,
            "active_key_index": self._keys.index(key) + 1,
            "total_keys": len(self._keys),
            "exhausted_pairs": len(self._exhausted),
            "total_pairs": len(self._pairs),
        }


_ROTATOR: _KeyModelRotator | None = None
_CLIENTS: dict[str, genai.Client] = {}


def _get_rotator() -> _KeyModelRotator:
    global _ROTATOR
    if _ROTATOR is None:
        _ROTATOR = _KeyModelRotator(_load_api_keys(), MODEL_CHAIN)
    return _ROTATOR


def _get_client(api_key: str) -> genai.Client:
    if api_key not in _CLIENTS:
        _CLIENTS[api_key] = genai.Client(api_key=api_key)
    return _CLIENTS[api_key]


def get_active_model_status() -> dict:
    """For the UI's active-model badge: which model/key is currently in
    use, and how much of the (key, model) rotation has been burned
    through this session."""
    return _get_rotator().status()


def _is_quota_error(exc: genai_errors.ClientError) -> bool:
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"


class _AllPairsExhausted(Exception):
    """Every (key, model) combination hit quota -- only reached after
    rotation is fully exhausted."""


class _TransientFailure(Exception):
    """Server overload or network blip -- worth retrying the same pair."""


def _call_once(prompt: str) -> str:
    rotator = _get_rotator()
    key, model = rotator.current()
    client = _get_client(key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"system_instruction": GENERATOR_SYSTEM_PROMPT, "temperature": 0.4},
        )
        return response.text.strip()
    except genai_errors.ClientError as exc:
        if not _is_quota_error(exc):
            raise  # bad request etc -- not something retrying/rotating fixes
        rotator.mark_exhausted(key, model)
        if rotator.advance() is not None:
            return _call_once(prompt)  # fresh (key, model) pair, try immediately
        raise _AllPairsExhausted(exc) from exc
    except (genai_errors.ServerError, httpx.TransportError, OSError) as exc:
        raise _TransientFailure(exc) from exc


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(_TransientFailure),
    wait=tenacity.wait_fixed(TRANSIENT_BACKOFF_SECONDS),
    stop=tenacity.stop_after_attempt(MAX_TRANSIENT_RETRIES),
    reraise=True,
)
def _call_with_transient_retry(prompt: str) -> str:
    return _call_once(prompt)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(_AllPairsExhausted),
    wait=tenacity.wait_random(RATE_LIMIT_WAIT_MIN_SECONDS, RATE_LIMIT_WAIT_MAX_SECONDS),
    stop=tenacity.stop_after_attempt(MAX_RATE_LIMIT_ATTEMPTS),
    reraise=True,
)
def _call_with_rate_limit_retry(prompt: str) -> str:
    """Last resort once every (key, model) pair has hit quota: reset the
    exhaustion tracking and wait out a short rate-limit window before
    trying again. If the quota is a genuine daily cap rather than a
    short-lived burst limit, this will still ultimately fail after
    MAX_RATE_LIMIT_ATTEMPTS -- there's nothing a wait of this length can
    do about a 24h cap, but it's cheap insurance against a shorter-lived
    per-minute limit."""
    _get_rotator().reset_exhaustion()
    return _call_with_transient_retry(prompt)


def generate_rewrite(original_resume_text: str, jd_text: str, feedback: str | None = None) -> str:
    """Call the Gemini generator to produce a rewritten resume, rotating
    across configured API keys and models on quota errors."""
    prompt_parts = [
        f"## Original resume:\n{original_resume_text}",
        f"## Target job description:\n{jd_text}",
    ]
    if feedback:
        prompt_parts.append(f"## Feedback on the previous attempt (address this):\n{feedback}")
    prompt = "\n\n".join(prompt_parts)

    try:
        try:
            return _call_with_transient_retry(prompt)
        except _AllPairsExhausted:
            return _call_with_rate_limit_retry(prompt)
    except (_AllPairsExhausted, _TransientFailure) as exc:
        # Unwrap back to the real google.genai error so existing callers
        # (e.g. app.py's `except genai_errors.ClientError/ServerError`)
        # keep working unchanged -- these two types are internal retry
        # bookkeeping, not something callers should need to know about.
        raise exc.__cause__ from None


@dataclass
class TruthfulnessResult:
    passed: bool
    unverified_terms: list[str]


def check_truthfulness(rewritten_text: str, original_resume_text: str) -> TruthfulnessResult:
    """Extract skill/tech-like terms from the rewritten text and verify
    every individual word in each term appears somewhere in the originally
    parsed resume text. Word-level (not exact-phrase) matching is
    deliberate: an LLM rewrite can legitimately reorder text or drop a
    connecting word like "and", which breaks contiguous-substring matching
    even though nothing was fabricated. A genuinely invented skill (e.g.
    "Kubernetes" never mentioned anywhere in the original) still won't be
    found as an individual word and is still flagged."""
    candidate_terms = extract_keywords(rewritten_text)
    original_words = set(_fold(original_resume_text).split())

    unverified = [
        term for term in candidate_terms
        if any(word not in original_words for word in _fold(term).split())
    ]
    return TruthfulnessResult(passed=len(unverified) == 0, unverified_terms=unverified)
