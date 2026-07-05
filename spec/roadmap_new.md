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
caching, or partial-resume across the whole pipeline.

## 1. Question generation: make the pipeline resumable

**Problem.** Each `generate` run reprocesses a flat topic list start-to-finish; a crash or
rate-limit failure partway through a large batch means re-running the whole command.

**Proposal.**

- Add a lightweight run-manifest (e.g. `.session/generate_{run}.toml`) that records
  per-topic status (pending/done/failed) so a re-run skips completed topics instead of
  redoing the whole batch. This does not require adopting a full DAG/workflow engine —
  a status file plus a "skip if already done" check gets most of the resumability value
  at a fraction of the complexity.
- If true DAG semantics (parallel independent topics, partial re-run of only failed
  nodes, cross-step caching) are actually wanted, treat that as a from-scratch design
  spike, not a refactor — no such structure exists today to build on.

## 2. Validation: cross-model consensus (optional, higher effort)

Ask two different providers to answer/judge the same question and flag disagreement,
as a stronger factual-correctness signal than a single model's self-assessment. Only
worth doing now that structured-output parsing and deterministic form checks are in
place, since consensus on unreliable parsing would just compound the fragility.

## 3. Exam-taking: incremental hardening

- Decompose `interactive_question_and_answer` in `take_exam.py` — it currently mixes
  scoring, session persistence, UI calls, and sound-hook stubs in one function;
  splitting scoring/persistence out from presentation would make both easier to test
  and to reuse from a future Textual/Web frontend.
- Surface the confidence-interval/pass-rate statistics more prominently in results
  (currently computed but not obviously shown) — this is a genuine differentiator
  worth using.

## 4. Frontend architecture: Textual TUI

Textual TUI genuinely isn't implemented (`spec/tui_frontend_plan.md` is a plan, no
`textual_ui.py` exists). Build it against the existing plan when a terminal-native UI
richer than the Rich CLI is wanted; re-add the `textual` dependency at that point.
