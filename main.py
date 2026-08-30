#!/usr/bin/env python
"""CLI entry point for resume-tailor-hybrid.

  candidate: score + rewrite a single resume against a job description.
  employer:  rank multiple resumes against one job description.
"""

from __future__ import annotations

import argparse

from src.agent_loop import DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_SCORE
from src.modes import candidate_mode, employer_mode


def _read_jd(jd_arg: str) -> str:
    """jd_arg is either a path to a .txt file, or the JD text itself."""
    if jd_arg.lower().endswith(".txt"):
        with open(jd_arg, "r", encoding="utf-8") as f:
            return f.read()
    return jd_arg


def run_candidate(args: argparse.Namespace) -> None:
    jd_text = _read_jd(args.jd)
    result = candidate_mode(args.resume, jd_text, args.target_score, args.max_iterations)

    print(f"Initial score: {result.initial_score}")
    print(f"Final score:   {result.final_score}")
    print(f"Target hit:    {result.hit_target}  (after {result.iterations_run} iteration(s))")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result.final_resume_text)
        print(f"\nFinal resume written to {args.output}")
    else:
        print("\n--- Final resume ---")
        print(result.final_resume_text)


def run_employer(args: argparse.Namespace) -> None:
    jd_text = _read_jd(args.jd)
    rankings = employer_mode(args.resumes, jd_text)

    print(f"{'Rank':<5}{'Score':<8}Resume")
    for i, r in enumerate(rankings, start=1):
        label = r.resume_path if not r.error else f"{r.resume_path} (ERROR: {r.error})"
        print(f"{i:<5}{r.score:<8}{label}")

    print("\n--- Gap summaries ---")
    for r in rankings:
        if r.error:
            continue
        print(f"\n{r.resume_path} (score {r.score}):")
        print(f"  matched: {', '.join(r.matched_keywords) or '(none)'}")
        print(f"  missing: {', '.join(r.missing_keywords) or '(none)'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resume-tailor", description="Resume tailoring agent")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    candidate_parser = subparsers.add_parser("candidate", help="Score and rewrite a single resume")
    candidate_parser.add_argument("--resume", required=True, help="Path to the candidate's resume PDF")
    candidate_parser.add_argument("--jd", required=True, help="Job description text, or path to a .txt file")
    candidate_parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    candidate_parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    candidate_parser.add_argument("--output", help="Path to write the final rewritten resume text")
    candidate_parser.set_defaults(func=run_candidate)

    employer_parser = subparsers.add_parser("employer", help="Rank multiple resumes against one job description")
    employer_parser.add_argument("--resumes", required=True, nargs="+", help="Paths to candidate resume PDFs")
    employer_parser.add_argument("--jd", required=True, help="Job description text, or path to a .txt file")
    employer_parser.set_defaults(func=run_employer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
