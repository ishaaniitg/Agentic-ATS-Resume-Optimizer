# Resume Tailor Agent — Hybrid Build Plan

## Overview

An agentic resume-tailoring system that parses a resume (PDF, including scanned/OCR
cases), scores it against a target job description, autonomously rewrites weak areas
while staying factually grounded in the original resume, and loops until it clears a
target score or hits a max-iteration cap. Includes both a candidate-facing mode and an
employer-facing screening mode.

Built as a synthesis of patterns from open-source references — not a fork of
any of them. Source repos are permissively licensed (Apache-2.0 / MIT); credit
all three in the final README.

## Goals (v1 scope — exactly these 4, nothing more)

- [ ] Parse resume in PDF format, with OCR fallback for scanned/image resumes
- [ ] Two modes: employer (screen/rank multiple resumes against one JD) and
      candidate (score + rewrite a single resume)
- [ ] Autonomously rewrite weak resume content to match the JD, while staying
      truthful to the source resume (no fabricated skills/experience)
- [ ] Re-score after rewriting, loop until a configurable target score is hit or a
      max-iteration cap is reached — keep the best-scoring version across all
      iterations, not just the last one

## Reference repos

| Repo | License | What we're borrowing |
|---|---|---|
| `sunithalv/ATS-Crewai` | Apache-2.0 | PDF parsing approach (PyMuPDF), employer/candidate mode split, CrewAI orchestration pattern |
| `Soroush-aali-bagi/resume-tailor-agents` | MIT | Generator + Judge agent split, per-criterion evaluation, dual-enforced truthfulness (prompt constraint + independent judge check), iterate-until-approved loop structure |
| `KryssSampi/cv-ats-pro-maker` | MIT | Configurable target-score + max-iteration loop design, "keep best version across iterations" logic (reference only — this repo is immature, don't copy code directly, just the loop design) |

Note on Phases 1 and 4: even with ATS-Crewai available as a reference, neither
piece strictly depends on it. PDF/OCR parsing (PyMuPDF + pytesseract) is a
standard, well-documented library pattern, not something specific to that
project. Employer/candidate mode is just two entry points wrapping the same
Phase 1-3 pipeline. Treat ATS-Crewai as a helpful working example of both, not
a hard dependency — if it becomes unreachable again mid-build, Phases 1 and 4
can proceed from library docs alone.

⚠️ Before copying any code from `resume-tailor-agents`: that repo has a raw `.env`
file committed to its root instead of `.env.example`. Check it isn't a real leaked
key before referencing anything from that repo, and make sure our own `.gitignore`
excludes `.env` from commit 1.

## Architecture (build in this order — each phase depends on the last)

### Phase 1 — Parsing layer (PDF + OCR)
- Extract text with PyMuPDF (`fitz`) first
- If extracted text is empty/near-empty (scanned resume), fall back to
  `pdf2image` + `pytesseract` OCR
- Output: normalized plain-text resume, ready for scoring
- Also need a simple JD input (paste text — skip URL scraping for v1)

### Phase 2 — Scoring function
- Local keyword/embedding match between resume text and JD (spaCy or
  sentence-transformers) — no LLM call needed here, deterministic and fast
- Output: numeric score + list of present/missing keywords
- Build and test this standalone before wiring into the agent loop

### Phase 3 — Rewrite loop with truthfulness (core piece — most time here)
- Generator agent: rewrites resume sections to better match the JD
- Judge/evaluator: re-scores using Phase 2's scorer, evaluates against criteria
- Truthfulness check: after each rewrite, extract skill/tool/tech tokens from the
  new text and verify each appears in the originally parsed resume text; reject
  and retry any rewrite that introduces unverified claims
- Loop control: configurable `target_score` and `max_iterations`; keep the
  best-scoring version seen across all iterations
- Log before/after per section — this becomes free groundwork for a future
  diff+rationale feature, even though that's not in v1 scope

### Phase 4 — Employer / candidate mode
- Candidate mode: single resume → Phase 1-3 pipeline → final resume + score
- Employer mode: N resumes + one JD → run Phase 1-2 on each → rank by score →
  return ranked list with each candidate's gap summary
- Skip auto-email sending (Gmail SMTP) for v1 — not in the 4 goals, adds setup
  overhead for no credit toward the required features

## Explicitly out of scope for v1

- Diff + rationale UI (nice-to-have; Phase 3's logging makes this cheap to add later)
- Job-description-from-URL scraping (Firecrawl or similar) — plain-text paste is
  enough for now
- Automated email dispatch to candidates

## Tech stack

- Python
- PyMuPDF, pdf2image, pytesseract (parsing)
- spaCy or sentence-transformers (scoring)
- An LLM API (OpenAI or Claude) for the Generator agent
- Simple CLI or Streamlit UI (optional, only if time allows)

## Rough time estimate

- Phase 1: ~2-3 hrs
- Phase 2: ~2-3 hrs
- Phase 3: ~4-6 hrs (spend the most time here — this is the differentiating piece)
- Phase 4: ~2 hrs

Total: ~1.5-2 focused days for all 4 goals.
