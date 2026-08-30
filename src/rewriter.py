"""LLM-backed resume rewriting (Gemini) plus a deterministic truthfulness
check that rejects any rewrite introducing skills/tech not present in the
originally parsed resume."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from src.scorer import extract_keywords, _fold

load_dotenv()

_CLIENT = None

GENERATOR_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_API_RETRIES = 3
API_RETRY_BACKOFF_SECONDS = 5

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


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env)")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


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


def generate_rewrite(original_resume_text: str, jd_text: str, feedback: str | None = None) -> str:
    """Call the Gemini generator to produce a rewritten resume."""
    prompt_parts = [
        f"## Original resume:\n{original_resume_text}",
        f"## Target job description:\n{jd_text}",
    ]
    if feedback:
        prompt_parts.append(f"## Feedback on the previous attempt (address this):\n{feedback}")

    prompt = "\n\n".join(prompt_parts)

    client = _get_client()
    last_error = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GENERATOR_MODEL,
                contents=prompt,
                config={"system_instruction": GENERATOR_SYSTEM_PROMPT, "temperature": 0.4},
            )
            return response.text.strip()
        except (genai_errors.ServerError, httpx.TransportError, OSError) as exc:
            # Transient (server overload, network blip) -- worth a retry.
            last_error = exc
            if attempt < MAX_API_RETRIES:
                time.sleep(API_RETRY_BACKOFF_SECONDS * attempt)
        except genai_errors.ClientError as exc:
            # Client-side (bad request, or 429 quota exhaustion) --
            # retrying immediately won't help; let the caller decide.
            raise
    raise last_error
