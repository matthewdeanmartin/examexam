# examexam

A CLI for **creating**, **validating**, **converting**, and **taking** multiple‑choice practice exams. Keep everything local as TOML question banks, generate new questions with an LLM, sanity‑check them, and study in a `rich` terminal UI.

---

## Install

### Recommended: `pipx`

```bash
pipx install examexam
```

> `pipx` keeps tools isolated and on your path. If you don’t use pipx, you can install with `python -m pip install examexam` or `uv tool install examexam`.

---

## Set up keys & environment

`examexam` reads a `.env` file (via `python-dotenv`) so you can keep secrets out of your shell history.

Create a `.env` next to where you run the commands:

```dotenv
# Use whichever providers you have
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...

# Optional: default model name your setup understands
EXAMEXAM_DEFAULT_MODEL=gpt4
```

> You’ll select a model with `--model` on commands that talk to an LLM (e.g., `gpt4`, `claude`, `gemini-1.5-pro`).

---

## Quick start

1. **Make a topics file** (one topic per line):

```
VPC
S3
ECS
IAM
RDS
```

2. **Generate questions** into a TOML bank (adds/extends the file):

```bash
examexam generate \
  --exam-name "AWS Associate Practice" \
  --toc-file topics.txt \
  --output-file data/aws-practice.toml \
  -n 10 \
  --model gpt4
```

3. **Validate** questions (model answers + Good/Bad triage):

```bash
examexam validate \
  --question-file data/aws-practice.toml \
  --exam-name "AWS Associate Practice" \
  --model claude
```

4. **Take the exam** in your terminal (resume supported):

```bash
examexam take --question-file data/aws-practice.toml
```

Optional: **convert** to pretty Markdown + HTML study notes:

```bash
examexam convert \
  --input-file data/aws-practice.toml \
  --output-base-name aws-practice
```

---

## What each command does (at a glance)

* **generate** — creates new multiple‑choice questions for your topics and appends them to a TOML question bank.
* **validate** — asks a model to answer each question and flags questions as Good/Bad with a short rationale, then saves that info back into the same file.
* **take** — launches a clean, keyboard‑only test UI with shuffling, per‑option explanations, progress, time stats, and resume.
* **convert** — turns a question bank into Markdown and HTML for easy reading.

---

## Where files go

* **Question banks (TOML):** store them wherever you like; many people use a `data/` folder (e.g., `data/aws-practice.toml`).
* **Session files:** saved automatically to `.session/<test-name>.toml` so you can quit and resume later.
* **Converted outputs:** `convert` writes `<base>.md` and `<base>.html` alongside your working directory.

> You don’t need to know the internal TOML schema—just point commands at your `.toml` files.

---

## Command reference

```text
examexam generate --exam-name <str> --toc-file <path> --output-file <path> [-n 5] [--model <name>]
examexam validate --question-file <path> --exam-name <str> [--model <name>]
examexam convert  --input-file <path> --output-base-name <str>
examexam take     --question-file <path>

# Global
--verbose   Enable detailed logging
```

---

## Tips & FAQ

* **Keys not picked up?** Ensure your `.env` sits in the directory where you run `examexam` and contains the correct provider key for the `--model` you chose.
* **Appending more questions:** Re‑run `generate` with the same `--output-file`.
* **Stuck session?** Delete `.session/<test-name>.toml` to start fresh.
* **Model naming:** Use names your environment supports (e.g., `gpt4`, `claude`, `gemini-1.5-pro`).

---

## Credits

* **Author:** Matthew Dean Martin (matthewdeanmartin)
* Thanks to **OpenAI** and **Google Gemini** models used during generation/validation.

## License

MIT License