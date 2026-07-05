# Modernization Roadmap

This document proposes strategic, architectural work — new capability, restructuring, or
design changes. For small fixes, dead code removal, and one-off cleanups, see
`spec/chores_bugs.md` instead.

## Context

ExamExam generates AWS-cert-style practice exams via LLM prompting, then lets a user
review/take them. The pipeline is currently described informally as a "DAG workflow,"
but it is actually a **linear sequence of independent CLI subcommands**
(`study-plan` → `research` → `generate` → `validate` → `convert` → `take`), each a
separate process invocation with no shared execution graph, dependency tracking,
caching, or partial-resume across the whole pipeline. The three areas called out by the
user — generation, validation of LLM output quality, and exam-taking — map to
`generate_questions.py`, `validate_questions.py`, and `take_exam.py` respectively.

## 1. Question generation: make the pipeline resumable and provider-honest

**Problem.** Each `generate` run reprocesses a flat topic list start-to-finish; a crash or
rate-limit failure partway through a large batch means re-running the whole command
(with `save_toml_to_file` re-reading/rewriting the entire output file per topic — an
O(n²) I/O pattern that also isn't crash-safe, since it's read-then-overwrite with no
temp file). Separately, three of the eight declared providers (Bedrock/Mistral/Cohere/
Meta/AI21, all routed through `BedrockCaller`) **do not call any real API** — they
return a hardcoded mock string. Anyone picking those providers today silently gets
fabricated exam content with no error.

**Proposal.**
- Add a lightweight run-manifest (e.g. `.session/generate_{run}.toml`) that records
  per-topic status (pending/done/failed) so a re-run skips completed topics instead of
  redoing the whole batch. This does not require adopting a full DAG/workflow engine —
  a status file plus a "skip if already done" check gets most of the resumability value
  at a fraction of the complexity.
- Switch `save_toml_to_file` to append-only writes (or write-temp-then-atomic-rename)
  instead of full read/rewrite per topic.
- Either implement real Bedrock calls (boto3 is already a dependency) or remove the
  mocked provider entries and their dependencies (`boto3`, `mistralai`) until real
  support exists — do not ship a code path that silently fabricates content.
- If true DAG semantics (parallel independent topics, partial re-run of only failed
  nodes, cross-step caching) are actually wanted, treat that as a from-scratch design
  spike, not a refactor — no such structure exists today to build on.

## 2. Validation: add deterministic checks alongside the LLM opinions

**Problem.** "Validation" currently means two LLM calls per question: ask the LLM to
answer it (`ask_llm`), and ask the LLM to judge it good/bad (`ask_if_bad_question`).
Both are parsed with fragile hand-rolled string/regex matching (`parse_answer`,
`parse_good_bad`) — the latter just checks whether the substring `"good"` appears
after a delimiter, so an explanation like *"not a good example, verdict: Bad"* would
misclassify as good. There is no deterministic verification of the *form* of a
question (the user's stated concern) — nothing in code checks that the generator
actually avoided "all of the above," that the correct-answer count matches what the
question text implies, or that a question isn't a near-duplicate of another already in
the bank.

**Proposal, roughly in order of cost/value:**
- **Deterministic schema/form checks** (cheap, high value, no LLM call needed):
  - Exact count of `is_correct` options matches what the question stem says
    ("select two" → exactly 2 correct options).
  - Reject banned patterns in option text (e.g. "all of the above", "none of the
    above", duplicate option text within a question).
  - Duplicate/near-duplicate question detection across the bank (simple text
    similarity is enough to start — e.g. normalized string match or a cheap
    embedding-distance check).
  - These become fast, free, deterministic gates that run before any LLM validation
    call, and their pass/fail should be recorded distinctly from the LLM's subjective
    good/bad verdict rather than folded into the same field.
- **Harden the existing LLM-based checks:**
  - Replace the ad hoc `parse_answer`/`parse_good_bad` string parsing with structured
    output (ask the LLM to return TOML/JSON matching a small schema, the same pattern
    already used successfully in `generate_questions.py`'s `extract_questions_toml` +
    retry-on-parse-failure loop). This removes an entire class of misclassification
    bugs and gives you a validated "verdict" enum instead of substring sniffing.
  - Currently a parse failure in `ask_llm` is silently downgraded to an empty answer
    list (`validate_questions_now`) — surface this as a distinct "validation
    inconclusive" state instead of scoring it as a wrong answer.
- **Optional, higher effort:** cross-model consensus (ask two different providers to
  answer/judge and flag disagreement) as a stronger factual-correctness signal than a
  single model's self-assessment. Only worth doing after structured-output parsing
  lands, since consensus on unreliable parsing just compounds the fragility.

## 3. Exam-taking: incremental hardening, not a rewrite

This module (`take_exam.py`) is the most solid part of the codebase — resumable
sessions, machine-mode answer providers for regression testing, and real statistics
(normal-approximation and exact Clopper-Pearson confidence intervals, a binomial
pass-rate test). Keep the design; a few targeted improvements:

- Score by stable option **id**, not option **text**, in `build_answer_feedback` —
  matching by string means two options with identical text in the same question (or
  minor regeneration-induced wording drift) produce undefined scoring behavior.
- Decompose `interactive_question_and_answer` — it currently mixes scoring, session
  persistence, UI calls, and sound-hook stubs in one function; splitting scoring/
  persistence out from presentation would make both easier to test and to reuse from
  a future Textual/Web frontend.
- Surface the confidence-interval/pass-rate statistics more prominently in results
  (currently computed but not obviously shown) — this is a genuine differentiator
  worth using.

## 4. Frontend architecture: finish what's declared, fix what's stale

The Protocol-based `FrontendUI` abstraction (`ui_protocol.py`) is a strong design
decision — business logic already doesn't touch UI directly. Two follow-ups:

- `spec/architecture.md` says the Web frontend is "Planned," but `web_ui.py` (769
  lines, FastAPI-backed) is already implemented and wired into `__main__.py`. Update
  the doc to match reality before it misleads the next contributor.
- Textual TUI genuinely isn't implemented (`spec/tui_frontend_plan.md` is a plan, no
  `textual_ui.py` exists) despite `textual>=1.0.0` being a hard dependency today. Either
  build it against the existing plan or drop the dependency until it's built — don't
  carry unused SDK weight for a frontend that doesn't exist yet.

## 5. Provider/router layer: stop killing the process from library code

`Router.call()` and `Router._get_caller()` call `sys.exit()` directly on certain
failure paths. This means any embedder of this code (tests, the web UI, a future
integration) can have its whole process terminated instead of catching an exception.
This should become a normal exception (there's already `FatalConversationError` in
`apis/types.py` — use it here) before any of the other work above tries to add new
test coverage around provider dispatch, since `sys.exit()` calls are why that logic is
essentially untestable today.

## Suggested sequencing

1. Router exception fix (#5) — unblocks testing everything downstream.
2. Bedrock/Mistral mock removal or real implementation (#1) — correctness/honesty issue.
3. Deterministic validation checks (#2) — directly answers the user's stated concern
   about bad question form, and is independent of the LLM-parsing hardening.
4. Structured-output validation parsing (#2) — larger, do after the cheap checks prove
   the deterministic-gate pattern works.
5. Generation resumability (#1) — nice-to-have once correctness issues are settled.
6. Exam-taking hardening (#3) and frontend doc/dependency cleanup (#4) — lower urgency,
   can happen opportunistically.
