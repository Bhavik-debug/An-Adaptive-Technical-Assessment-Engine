# The report screen (Day 28)

`/interviews/:sessionId/report`, signed-in only. It shows the report the
backend built (Day 25, [reporting.md](reporting.md)) and nothing else: every
number on the page comes from `GET /api/sessions/{id}/report`. The page
computes none of them and never fills a gap with an estimate.

Code: `frontend/src/pages/report/ReportPage.tsx`, `src/lib/report.ts` (reading
the payload), `src/components/SkillRow.tsx` (the skill bar, shared with the
dashboard's skill overview). Tests: `src/pages/report/report.test.tsx`.

## 1. Requests and states

1. `GET /api/sessions/{id}`: role, level, mode, topics, state, finish time.
   **404** (no such session, or another account's) → "Report not found". The
   page never says which.
2. If the session is not in `REPORTING`/`COMPLETED` → "This interview is not
   finished yet" with **Continue the interview**. The report is not requested.
3. Otherwise `GET /api/sessions/{id}/report`, polled every 2 s (the existing
   `usePolling` hook, 90 s ceiling):

| Response | Screen |
|---|---|
| 202 `pending`/`processing` | "Preparing your report…" (`role=status`). After 90 s: "still being prepared" and **Check again**. The job keeps running server-side. |
| 200 `ready` | The report (§2). The session details are re-read once, because a ready report always belongs to a completed session, so the header switches from "Finishing" to "Completed". |
| 500 with body `status: "failed"` | "This report could not be prepared". The answers and grades are saved, and an operator can re-queue it (`requeue_failed_report`). The error code is not shown. |
| 409 | As "not finished" (a race with the session state). |
| other error / network | The server's message and **Try again** |
| 200 but `stats` unusable | "This report could not be displayed". Nothing is estimated in its place. |

To tell "failed" from any other 500, `ApiError` now keeps the parsed error
body (`ApiError.body`). Nothing else in the client changed.

Reloading the page re-runs the same requests, so the state shown is always the
server's.

## 2. What is shown, and from which field

| Section | Fields (`stats.*` unless noted) |
|---|---|
| Header | session `role`, `level`, `mode`, `topic_keys`, `state`, `ended_at` |
| Summary | `counts.primary_questions`, `counts.follow_ups_asked`, `generated_at`; `narrative.summary` labelled by `narrative_source` (`model`: "Written by an AI model… it did not produce any of the numbers"; `template`: "A standard summary"); `counts.grading_failures` > 0 → a warning that those answers are not counted |
| Results by topic | `topics[].observed.display` / `interval` / `insufficient_evidence`, `topics[].questions_asked`; within each topic, the **observed** `subtopics[]` the same way, with `n_observations` as "answers". Unmeasured or too-uncertain (RD > 0.8) skills get a label ("Not assessed" / "Not enough evidence yet"), never a bar. Every bar shows its interval (plan §9.3). |
| Where you stand | `gaps` (ranked, "Areas to improve"), `strengths`, `needs_more_assessment`. All are backend lists, judged against the chosen level. |
| Feedback | `narrative.strengths` / `growth_areas` / `next_steps`, only when `narrative_source` is `model`, and only the non-empty lists. The template's lists only restate the three lists above, so they are not repeated. |
| Question by question | `questions[].order`, `topic_key`, `subtopic_key`, `score` (0–1, shown as `round(score × 100) / 100`), `follow_ups[].score`. "Not graded" for a missing grade. The half-weight follow-up caveat (from `methodology_notes`) is stated. |
| How to read this report | A plain-language reading of the scale and interval; "automated grader still being validated against human reviewers" (Phase 4 is acceptance-pending); the resume-prior note when `priors.source` is `resume_profile`; `methodology_notes` in a collapsible "Method notes" |

Names come from `GET /api/setup-options` (`display_name`), with a readable
fallback for unknown keys.

**Not shown, on purpose:** `theta`, `rd`, `target_theta`, `stop_reasons`,
`question_id`, `log_seq`, `priors.seeded_subtopics`, `narrative_error`, and
`error`. They are internal or diagnostic, and `lib/report.ts` does not copy
them into the view, so no component can render them by accident. A test
asserts that the page text contains none of them.

Showing per-answer grades **after** the interview is consistent with the
interview screen, which drops them only **during** it
([interview-screen.md](interview-screen.md) §2), and with the plan's
"per-question evidence cards".

## 3. Plan items not met, and why

Plan §3, Day 28: "overall band, per-skill bars with confidence intervals,
per-question evidence cards, ranked gaps, 2-week study plan. Live skill graph
during the session (Recharts)."

| Item | Status |
|---|---|
| Per-skill bars with intervals | done |
| Ranked gaps | done (`gaps` is ranked by the backend) |
| Per-question cards | **partial**: grade, topic and subtopic. The report has no question text and no grader evidence (quotes or concepts), so there is no "evidence" to show. Adding it is a backend contract decision. Concept keys are deliberately kept from candidates today. |
| Overall band | **not shown**: the report has no overall field, and the narrative prompt forbids an overall score. The page does not compute one. |
| 2-week study plan | **not shown**: no such field. `narrative.next_steps` (model-written, ≤ 5 items) is shown as "Next steps". |
| Live skill graph during the session | **not built**: the session snapshot and stream deliberately carry no θ or display values (Day 27), so there is nothing real to plot. This needs a backend decision about what may be shown mid-interview. Recharts was not added. |

## 4. Accessibility and layout

Headings h1 → h2 → h3 in order; bars are `role="img"` with the full reading
("Graphs: 48 out of 100, likely range 30 to 66"); lists are real lists, with
gaps as an ordered list; loading and pending messages use `role="status"`;
"Method notes" is a native `<details>`. Single column below 640 px, three-column
lists from `sm`. Checked in a real browser (headless Edge) at 1280 px and
375 px with no horizontal scroll, and keyboard-only to **Back to dashboard**
and the Method notes toggle.

## 5. Visual design (Day 28, follow-up)

Charts are plain markup and inline SVG in `pages/report/ReportVisuals.tsx`.
No chart library was added. Every mark is placed from a backend number;
nothing is computed except its position on the 0–100 axis.

| Section | Form |
|---|---|
| Header | gradient hero card: role, level/mode, status, finish time, topic chips |
| Stat tiles | `counts.primary_questions`, `follow_ups_asked`, `answers_graded`, and "Skills measured" (how many subtopics the backend rated: observed and RD ≤ 0.8) |
| Results by topic | a semicircle meter per topic: the estimate dot on its likely-range arc, the number in the middle; "Not assessed" / "Not enough evidence yet" instead of a number |
| Skill estimates | one 0–100 axis, a row per **rated** skill, sorted by estimate: the dot is the estimate, the band the interval, the value printed at the end; a hover tooltip; legend; **"Show as a table"** with every value |
| Where you stand | three tinted panels (areas to improve, strengths, needs more assessment) with icons and counts |
| Feedback | model-written sections only, with icons |
| Question by question | a column chart (answer grade, plus lighter follow-up columns; "not graded" stub) above the list, which has a small meter per grade |

**Colour has one job:** the backend's classification of a rated skill.
Strength is `--viz-strength` (#036639), area to improve `--viz-gap`
(#c98a10), any other rated skill `--viz-measured` (#2559bf, the primary).
Topic meters and grade columns use the primary only. The three were run
through the dataviz palette validator against the card surface: lightness,
chroma, CVD separation (worst protan ΔE 18.3) and normal-vision floor all
pass. The app's original `--warning` amber **failed** CVD separation against
green (protan ΔE 4.0), which is why a lighter amber was chosen. `--viz-gap` is
2.87:1 against white, so it is never used for text, and every amber mark also
carries an icon, a text label, a printed value and a table row. Text always
uses text colours.

**Accessibility:** the meters and skill rows are `role="img"` with the full
reading as their name ("Graphs: 24 out of 100, likely range 12 to 41"). Each
chart column has an `aria-label` with its grades. Tooltips are hover-only
conveniences, never the only way to a value. The marks are not focusable
(focusable non-interactive elements fail `jsx-a11y`); keyboard and
screen-reader users get the same values from names, printed numbers and the
table. Panel counts are `aria-hidden` (the list conveys its length).

Verified in headless Edge: the real journey on the isolated stack (20/20),
and a layout-only render of a fuller report (the unit-test fixture shape,
served by route interception, not product data) at 1280 px and 375 px, with
no horizontal scroll.

