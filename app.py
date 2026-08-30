"""Streamlit UI: a thin manual-testing surface over the existing Phase 1-4
pipeline. Every button below calls the existing src/* functions directly --
no parsing, scoring, rewriting, or ranking logic is reimplemented here.

Entry points used (see PLAN.md phases 1-4):
  - src.parser.parse_resume          (Phase 1)
  - src.scorer.score_resume          (Phase 2)
  - src.agent_loop.run_rewrite_loop  (Phase 3 -- used directly, not via
    src.modes.candidate_mode, because the UI needs the per-iteration
    history/changed_sections that candidate_mode discards)
  - src.modes.employer_mode          (Phase 4, employer mode wrapper)
"""

import os
import tempfile

import streamlit as st
from google.genai import errors as genai_errors

from src.agent_loop import DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_SCORE, run_rewrite_loop
from src.modes import employer_mode
from src.parser import ParseError, parse_resume
from src.scorer import score_resume

st.set_page_config(page_title="Resume Tailor -- Manual Test UI", layout="wide")
st.title("Resume Tailor -- Manual Test UI")
st.caption(
    "Thin debugging surface over src/parser.py, src/scorer.py, src/agent_loop.py "
    "and src/modes.py. No pipeline logic lives in this file."
)

mode = st.radio("Mode", ["Candidate", "Employer"], horizontal=True)


def _save_upload_to_temp(uploaded_file) -> str:
    """parse_resume() takes a file path, not bytes -- save the upload to a
    temp file so we can call it unmodified."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


if mode == "Candidate":
    st.header("Candidate mode")

    resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Job description", height=150)

    col1, col2 = st.columns(2)
    with col1:
        target_score = st.number_input(
            "Target score", value=DEFAULT_TARGET_SCORE, min_value=0.0, max_value=100.0
        )
    with col2:
        max_iterations = st.number_input(
            "Max iterations", value=DEFAULT_MAX_ITERATIONS, min_value=1, step=1
        )

    if st.button("Analyze", disabled=not (resume_file and jd_text)):
        try:
            resume_path = _save_upload_to_temp(resume_file)
            resume_text = parse_resume(resume_path)
            st.session_state["resume_text"] = resume_text
            st.session_state["jd_text"] = jd_text
            st.session_state["score_result"] = score_resume(resume_text, jd_text)
            st.session_state.pop("loop_result", None)  # stale after re-analyze
        except (ParseError, ValueError) as exc:
            st.error(f"Parsing failed: {exc}")

    if "resume_text" in st.session_state:
        st.subheader("Phase 1 -- parsed resume text")
        st.text_area("Extracted text", st.session_state["resume_text"], height=250)

        st.subheader("Phase 2 -- score")
        result = st.session_state["score_result"]
        st.metric("Score", f"{result.score}%")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.write("**Matched keywords**")
            st.write(result.matched_keywords or "(none)")
        with gcol2:
            st.write("**Missing keywords**")
            st.write(result.missing_keywords or "(none)")

        st.subheader("Phase 3 -- rewrite + truthfulness loop")
        if st.button("Rewrite"):
            try:
                with st.spinner("Running generator -> truthfulness check -> judge loop..."):
                    st.session_state["loop_result"] = run_rewrite_loop(
                        st.session_state["resume_text"],
                        st.session_state["jd_text"],
                        target_score=target_score,
                        max_iterations=int(max_iterations),
                    )
            except genai_errors.ClientError as exc:
                st.error(f"Gemini API error (not a pipeline bug): {exc}")
            except genai_errors.ServerError as exc:
                st.error(f"Gemini API unavailable after retries (not a pipeline bug): {exc}")

        if "loop_result" in st.session_state:
            loop_result = st.session_state["loop_result"]

            st.write("**Per-iteration log**")
            for log in loop_result.history:
                if log.iteration == 0:
                    st.markdown(f"- Iteration 0 (initial parse) -- score: {log.score}")
                    continue

                status = "PASSED" if log.accepted else f"REJECTED ({log.rejection_reason})"
                with st.expander(f"Iteration {log.iteration} -- score: {log.score} -- {status}"):
                    if log.accepted and log.changed_sections:
                        for section, (before, after) in log.changed_sections.items():
                            st.markdown(f"**Section: {section}**")
                            bcol, acol = st.columns(2)
                            with bcol:
                                st.text_area(
                                    "before", before, height=150,
                                    key=f"before-{log.iteration}-{section}",
                                )
                            with acol:
                                st.text_area(
                                    "after", after, height=150,
                                    key=f"after-{log.iteration}-{section}",
                                )
                    elif log.accepted:
                        st.caption(
                            "Accepted, but the section splitter found no per-section diff."
                        )
                    else:
                        st.caption(
                            "This draft was rejected by the truthfulness check and "
                            "discarded -- it was never scored."
                        )

            st.subheader("Final result")
            fcol1, fcol2, fcol3 = st.columns(3)
            fcol1.metric("Final score", f"{loop_result.best_score}%")
            fcol2.metric("Iterations run", loop_result.iterations_run)
            fcol3.metric("Hit target", "Yes" if loop_result.hit_target else "No")

            accepted_scores = {log.iteration: log.score for log in loop_result.history if log.accepted}
            best_iteration = max(accepted_scores, key=accepted_scores.get) if accepted_scores else 0
            if best_iteration != loop_result.iterations_run:
                st.info(
                    f"Best score ({loop_result.best_score}%) came from iteration "
                    f"{best_iteration}, not the last iteration run ({loop_result.iterations_run}). "
                    "The text below is that best-scoring version -- confirming the loop kept "
                    "it rather than defaulting to the final draft."
                )
            else:
                st.info(
                    f"Best score ({loop_result.best_score}%) is from iteration {best_iteration}, "
                    "which is also the last iteration run."
                )

            st.subheader("Best-scoring resume text")
            st.text_area("Final resume", loop_result.best_resume_text, height=300)

else:  # Employer
    st.header("Employer mode")

    resume_files = st.file_uploader("Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    jd_text = st.text_area("Job description", height=150)

    if st.button("Rank", disabled=not (resume_files and jd_text)):
        resume_paths = []
        name_by_path = {}
        for f in resume_files:
            path = _save_upload_to_temp(f)
            resume_paths.append(path)
            name_by_path[path] = f.name

        st.session_state["rankings"] = employer_mode(resume_paths, jd_text)
        st.session_state["name_by_path"] = name_by_path

    if "rankings" in st.session_state:
        rankings = st.session_state["rankings"]
        name_by_path = st.session_state["name_by_path"]

        st.subheader("Ranked candidates")
        st.table([
            {
                "Rank": i,
                "File": name_by_path.get(r.resume_path, r.resume_path),
                "Score": r.score,
                "Error": r.error or "",
            }
            for i, r in enumerate(rankings, start=1)
        ])

        st.subheader("Gap summaries")
        for r in rankings:
            label = name_by_path.get(r.resume_path, r.resume_path)
            with st.expander(f"{label} -- score {r.score}"):
                if r.error:
                    st.error(r.error)
                    continue
                st.write("**Matched:**", ", ".join(r.matched_keywords) or "(none)")
                st.write("**Missing:**", ", ".join(r.missing_keywords) or "(none)")
