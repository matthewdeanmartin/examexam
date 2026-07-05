# Hacking On Examexam

This page is for people changing the program itself. It is intentionally different from the generic contributor workflow page.

## High-level architecture

`examexam` has three main layers:

1. CLI orchestration in `examexam/__main__.py`
2. business logic in modules like `generate_questions.py`, `validate_questions.py`, and `take_exam.py`
3. adapters around UI frontends, prompt templates, and model providers

The entrypoint parses arguments, picks a frontend, resolves a model, and dispatches to a command-specific `*_now()` function.

## Core modules

### `examexam/__main__.py`

- defines the CLI
- selects `cli`, `gui`, `tui`, or `web` frontend
- sends commands into the business-logic entrypoints

One important special case: the web frontend only supports `take`, so `__main__.py` rejects other commands when `--frontend web` is selected.

### `examexam/take_exam.py`

This is the exam engine.

It owns:

- loading question banks
- selecting tests from `data/`
- saving and resuming `.session/*.toml`
- shuffling unanswered questions and option order
- collecting answers through the frontend protocol
- scoring, timing, and confidence interval reporting

The session file is the durable source of in-progress exam state.

### `examexam/generate_questions.py`

This is the question-generation pipeline.

It:

- reads the topic list
- renders the Jinja template for each topic
- calls the selected model through `Router`
- extracts TOML from the model output
- validates the result against the expected schema
- appends valid questions to the output bank atomically

The schema enforcement lives in `examexam/find_the_toml.py`.

### `examexam/validate_questions.py`

This is the review pass for a question bank.

It runs deterministic structural checks first, then asks an LLM to:

- answer each question
- judge whether the question is good or bad

Results are written back into the same TOML bank. Inconclusive LLM responses are recorded separately instead of silently counting as wrong.

### `examexam/jinja_management.py`

This is the prompt-template loader and deployer.

Template lookup order is:

1. `./prompts/` in the current working directory
2. development-mode prompts in the source tree
3. prompts bundled in the installed package

That makes prompt iteration easy without patching package files.

## Frontend model

The program avoids hard-coding Rich calls into the core logic by using a protocol in `examexam/ui_protocol.py`.

Current frontends:

- `frontends/rich_ui.py`: default terminal UI
- `frontends/tkinter_ui.py`: local desktop UI
- `frontends/web_ui.py`: localhost FastAPI app
- `frontends/manager_gui.py`: separate management GUI

`frontends/__init__.py` is the registry/factory that maps the `--frontend` flag to an implementation.

If you add a new frontend, the rule of thumb is:

- business logic should only depend on `FrontendUI`
- frontend-specific event loops stay in the frontend implementation
- command dispatch stays in `__main__.py`

## Data model

The persistent question-bank format is TOML, but the UI layer mostly works with typed dataclasses from `examexam/models.py` such as:

- `Question`
- `Option`
- `AnswerFeedback`
- `ExamResult`
- `TestInfo`

The code still moves some raw dictionaries around, especially at file and session boundaries, so a lot of the current architecture is "typed edge adapters around dict-heavy pipelines."

## Model provider routing

Provider selection lives in `examexam/apis/conversation_and_router.py`.

The main ideas:

- provider aliases such as `openai` and `anthropic` map to current default model ids
- `pick_model()` resolves `--model`, `--model-provider`, and `--model-class`
- `Router` lazily instantiates caller classes from `third_party_apis.py`
- fatal provider/model misconfiguration raises `FatalConversationError`

That fatal exception is important because the CLI can catch it and exit cleanly instead of letting a bad provider kill the process abruptly.

## Repo-level conventions

- `data/` is the conventional home for question banks
- `.session/` stores resumable exam state
- `study_guide/` stores generated markdown study output
- `prompts/` in the repo root overrides packaged prompt templates
- `spec/` holds design notes and future plans

## Where to start for common changes

- New CLI behavior: start in `examexam/__main__.py`
- Question-bank schema changes: start in `examexam/find_the_toml.py`
- Exam-taking behavior: start in `examexam/take_exam.py`
- Prompt wording or format constraints: start in `examexam/prompts/`
- Web UX changes: start in `examexam/frontends/web_ui.py` plus `frontends/web_templates/`
- New provider support: start in `examexam/apis/`
