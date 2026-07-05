# Chores & Bugs

Small, concrete, mostly independent fixes found during a codebase review. These are not
architectural — see `spec/roadmap_new.md` for the larger proposals. Roughly ordered by
risk/impact within each section.

## Bugs / correctness risks

- **`openai>0.28` dependency floor is wrong.** `third_party_apis.py` uses the v1+/v2.x
  client API (`openai.OpenAI(...)`, `client.chat.completions.create(...)`), which is a
  breaking change from the pre-1.0 API implied by `>0.28`. Anyone installing near the
  floor gets a hard `AttributeError` at runtime. Bump the floor to a real `>=1.x`
  constraint that matches the code.
- **Bedrock/Mistral/Cohere/Meta/AI21 providers are non-functional mocks.**
  `BedrockCaller` (`third_party_apis.py:147-165`) returns a hardcoded canned string for
  all five provider names instead of calling any real API — this fabricates exam
  content silently with no error surfaced to the user. Either implement it for real or
  remove the provider entries/dependencies until it's implemented.
- **`Router.call()` / `Router._get_caller()` call `sys.exit()` directly**
  (`conversation_and_router.py:96,166`) from inside library code, not the CLI
  entrypoint. Any caller (tests, web UI, future embedder) can have the whole process
  killed. Replace with raising `FatalConversationError` (already defined in
  `apis/types.py`) and let `__main__.py` decide how to exit.
- **Validation parse failures are silently downgraded.** In `validate_questions_now`,
  a parse failure in `ask_llm` (raises `ExamExamTypeError`) is caught and turned into an
  empty answer list, which then scores as simply "wrong" — this hides a parsing bug
  behind what looks like a normal wrong answer. Surface it as a distinct
  "validation inconclusive" state.
- **`parse_good_bad` misclassifies on substring match.** It checks whether the literal
  substring `"good"` appears in the LLM's response after a `---` delimiter — text like
  "not a good example, verdict: Bad" would misclassify as good. Needs a stricter parse
  (ideally structured output, see roadmap #2).
- **Scoring compares options by text, not by id.** `build_answer_feedback` in
  `take_exam.py` matches selected options against `is_correct` by option **text**
  equality. Two options with identical text in one question (or minor regeneration
  wording drift) → undefined scoring behavior. Score by stable option id instead.
- **`save_toml_to_file` is not crash-safe.** It reads the entire existing output file,
  appends, and overwrites — a crash mid-write can corrupt the file, and it's an O(n²)
  I/O pattern for large multi-topic generation runs. Switch to write-temp-then-rename
  (atomic) and avoid full re-read per topic.
- **`FakeBotCaller` has a built-in ~10% random failure rate** (`third_party_apis.py:183`)
  used by tests, making any test that exercises it inherently flaky. The one test that
  does (`tests/test_generate_questions.py`) only asserts `result == 0`, which doesn't
  actually verify successful generation. Either make the fake deterministic (seeded, or
  configurable failure rate defaulting to 0) or fix the test's assertions.

## Dead code / repo clutter

- `dead_code/take_exam_v1.py` — an old module kept in-tree (excluded from ruff) instead
  of relying on git history. Delete it; git log already preserves it.
- `apis/utilities.py::load_env()` is a no-op stub with the real implementation commented
  out, while every actual module calls `dotenv.load_dotenv()` independently at import
  time. Either make `load_env()` do the real thing and use it consistently, or delete
  the stub — right now it's a misleading dead abstraction.
- `take_exam.py::play_sound()` is a no-op placeholder with commented-out code. Remove or
  implement.
- Four separate test files for the same module with no clear scope boundary:
  `test_take_exam.py`, `test_take_exam_2.py`, `test_take_exam_3.py`,
  `test_take_exam_again.py`. Consolidate into one `test_take_exam.py` (and similarly
  `test_validate_questions.py` + `test_validate_questions_two.py`).
- Duplicated fatal-error detection logic between `generate_questions.py`
  (`_fatal_if_misconfigured`/`_is_fatal_message`) and `validate_questions.py`
  (`_fatal_precheck`/`_is_fatal_message`) — nearly identical marker lists maintained in
  two places. Extract to one shared helper.
- Duplicated TOML schema validation: `generate_questions.py::_validate_schema` and
  `find_the_toml.py::_valid_schema` implement almost the same checks independently
  (and have already drifted — one requires non-empty `question`, the other only checks
  it's a string). Extract to one shared validator.
- `GoogleCaller._initialize_google()` has a Colab-specific fallback
  (`from google.colab import userdata`) baked into production code outside any notebook
  context. Remove unless Colab is an actual supported deployment target.
- `SecureRandom` (`utils/secure_random.py`) is a misleading name — it's a deterministic
  seedable RNG wrapper for reproducible tests, not cryptographically secure. Rename to
  something like `SeededRandom` to avoid future misuse in an actually
  security-sensitive context.

## Dependency cleanup

- Both `rtoml` and `toml` are declared; only `rtoml` (aliased as `toml`) appears to be
  used. Drop the unused `toml` dependency after confirming.
- `orjson` is declared but no usage was found in the core modules reviewed — audit and
  drop if genuinely unused.
- `textual>=1.0.0` is a hard dependency for a TUI frontend that doesn't exist yet
  (`textual_ui.py` is not implemented, only planned). Either build it or drop the
  dependency until it's built.
- `mistralai>=1.6.0` and much of the `boto3` surface are unused given `BedrockCaller` is
  a mock — see the bug above; resolve together.
- Three formatting tools configured simultaneously (`[tool.black]`, `[tool.ruff]`,
  `[tool.isort]`), and `[tool.black] target-version = ['py39']` contradicts
  `requires-python = ">=3.11"` and ruff's `target-version = "py311"`. Pick one formatter
  (ruff format is the natural choice given ruff is already primary) and delete the
  others' config.
- `minimum_test_coverage = 35` in `pyproject.toml` is set low enough to make the low
  coverage in `__main__.py` and `manager_gui.py` (both 0%) invisible in CI. Raise it
  incrementally as coverage improves rather than leaving it as a permanent low bar.

## Docs

- `spec/architecture.md` lists the Web frontend as "Planned," but `web_ui.py` (769
  lines, FastAPI-backed) is already implemented and wired into `__main__.py`. Update the
  status table.
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
