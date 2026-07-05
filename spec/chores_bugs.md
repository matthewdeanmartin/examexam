# Chores & Bugs

Small, concrete, mostly independent fixes found during a codebase review. These are not
architectural — see `spec/roadmap_new.md` for the larger proposals. Roughly ordered by
risk/impact within each section.

## Bugs / correctness risks

- **`FakeBotCaller` has a built-in ~10% random failure rate** (`third_party_apis.py:183`)
  used by tests, making any test that exercises it inherently flaky. The one test that
  does (`tests/test_generate_questions.py`) only asserts `result == 0`, which doesn't
  actually verify successful generation. Either make the fake deterministic (seeded, or
  configurable failure rate defaulting to 0) or fix the test's assertions.

## Dead code / repo clutter

- `GoogleCaller._initialize_google()` has a Colab-specific fallback
  (`from google.colab import userdata`) baked into production code outside any notebook
  context. Remove unless Colab is an actual supported deployment target.
- `SecureRandom` (`utils/secure_random.py`) is a misleading name — it's a deterministic
  seedable RNG wrapper for reproducible tests, not cryptographically secure. Rename to
  something like `SeededRandom` to avoid future misuse in an actually
  security-sensitive context.

## Dependency cleanup

- `minimum_test_coverage = 35` in `pyproject.toml` is set low enough to make the low
  coverage in `__main__.py` and `manager_gui.py` (both 0%) invisible in CI. Raise it
  incrementally as coverage improves rather than leaving it as a permanent low bar.

## Docs

- No doc currently describes the generation pipeline's actual linear, non-DAG nature —
  worth a short note so contributors don't go looking for graph/workflow-engine code
  that doesn't exist.

## Test coverage gaps (lowest-effort wins first)

- `examexam/__main__.py` — 0% coverage; the entire CLI arg-parsing/dispatch layer is
  untested. Even a handful of subprocess/CLI-invocation smoke tests would catch
  regressions in argument wiring.
- `manager_gui.py` — largest file in the repo (1,324 lines), 0% coverage.
- `tkinter_ui.py`, `doctor.py`, `generate_pretty_study_plan.py` — also 0% coverage.
- `take_exam.py` sits at ~64% despite being the best-designed module — worth closing
  the gap given how much logic (scoring, resume, stats) lives there.
