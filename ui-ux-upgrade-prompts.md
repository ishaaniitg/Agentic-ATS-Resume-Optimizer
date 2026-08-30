# UI/UX Upgrade — Claude Code Prompt Library
### For: Resume Tailor -- AI ATS Optimizer (app.py)

Run these **one at a time, in order**, in separate Claude Code turns. Test the app after each one before moving to the next — this keeps regressions easy to spot and easy to revert (commit after each prompt lands cleanly).

General rule baked into every prompt: **never touch pipeline logic in `src/*.py`** — only `app.py` and new asset/CSS files. This matches how the app is already structured (app.py is a thin UI layer over parser/scorer/agent_loop/modes).

Optional libraries worth adding for a couple of these: `plotly` (for the score gauge) and `streamlit-extras` (nicer badges/cards) — mention these to Claude Code if you want it to use them instead of hand-rolled CSS.

---

## Prompt 1 — Design system foundation

```
Set up a proper design system for this Streamlit app (app.py) without touching any pipeline logic in src/.

- Add an assets/theme.css file with: a defined color palette (primary, background, surface, border, success/warning/error, muted text), a Google Font import (Inter or similar) applied globally, consistent border-radius and shadow tokens, and a spacing scale.
- Inject this CSS via st.markdown(unsafe_allow_html=True) at the top of app.py.
- Hide Streamlit's default hamburger menu and "Made with Streamlit" footer via CSS, but keep the native sidebar/toolbar functional.
- Replace the current hardcoded purple gradient header with a component driven by the theme file, so all colors are managed in one place going forward.
- Do not change any function signatures or logic in src/*.py.
```

## Prompt 2 — Layout & navigation restructure

```
Restructure the top-level layout of app.py for a cleaner information architecture:

- Move mode selection (Candidate / Employer) into the sidebar instead of inline radio buttons, alongside the model/quota status indicator ("Active Model: ...") shown as a compact sidebar status card.
- Use st.columns to give the main content area a consistent max-width and padding, centered on wide screens.
- Group each phase (Upload, Parsed Resume, Match Score, Rewrite Loop) into clearly separated st.container(border=True) sections with icons and section headers, replacing the current plain-text "1. UPLOAD & TARGET" style labels.
- Add a simple horizontal stepper at the top showing progress: Upload → Score → Rewrite.
```

## Prompt 3 — Upload & input experience

```
Improve the resume/job-description upload experience in app.py:

- Style the st.file_uploader as a clear dropzone that shows filename, size, and a success checkmark once parsing succeeds — and a specific, human-readable error state if parsing/OCR fails (not a generic exception).
- For Employer mode's multi-file upload, show uploaded resumes as removable chip/card elements with filename + size, not the current plain file list.
- Disable the Analyze/Rank button with a tooltip explaining what's missing (e.g. "Upload a resume and paste a job description") instead of just greying it out with no explanation.
- Add a live word/character counter under the job description textarea.
```

## Prompt 4 — Score & results visualization (Candidate mode)

```
Redesign the Candidate-mode results in app.py:

- Replace the plain "48.0%" number with a radial/gauge chart (use plotly) showing the ATS match score on a red→amber→green scale, with the target score marked as a line on the gauge.
- Show the "50% semantic + 50% keyword" breakdown as two small labeled bars instead of plain caption text.
- Redesign the matched/missing/conceptual-gap keyword pills with consistent sizing and spacing; put "Conceptual skill gaps" in its own collapsible section with a short tooltip explaining what it means (it's the least intuitive category to a first-time user).
- For the rewrite loop, add a side-by-side or diff-highlighted view of original vs. rewritten sections per iteration, with the score delta (+X%) shown per iteration, and a visual marker on any iteration where a rewrite was rejected by the truthfulness check.
```

## Prompt 5 — Employer mode table polish

```
Improve the Employer-mode ranked-candidates view in app.py:

- Replace the default table with a styled ranked list: rank badge, candidate name, inline score bar with percentage, and a status badge — using the design system from theme.css, not default Streamlit table borders.
- Make each row expandable into its gap summary, with matched/missing keyword pills styled consistently with Candidate mode.
- Add a way to sort by score or filter by a minimum score threshold above the table.
```

## Prompt 6 — Feedback, loading, and empty states

```
Add proper feedback states throughout app.py:

- Replace bare spinners with st.status showing the actual pipeline step in progress (e.g. "Parsing PDF...", "Scoring against job description...", "Running rewrite iteration 2 of 5...").
- Add a friendly empty state (icon + short text) for each results section before Analyze/Rank has been run, instead of blank space.
- Use st.toast for success/failure events (e.g. "Resume parsed successfully", "Gemini quota exceeded — try again shortly") in addition to inline st.error.
- Make sure the existing genai_errors try/except surfaces a clear, non-technical message with a retry option.
```

## Prompt 7 — Final polish & responsiveness pass

```
Do a final consistency and responsiveness pass on app.py and assets/theme.css:

- Check layout on a narrow viewport (~380px) and stack columns vertically where needed.
- Ensure consistent font sizes, spacing, and color usage across every section against the theme.css tokens — remove any leftover default Streamlit styling.
- Add a proper page title and favicon via st.set_page_config.
- Confirm no pipeline logic in src/*.py was touched during this entire styling pass — only app.py and asset/CSS files changed.
```
