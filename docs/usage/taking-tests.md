# Taking Tests

This page is for people who already have a TOML question bank and want to study with it.

## Basic flow

If your question file is `data/aws_exam.toml`:

```bash
examexam take --question-file data/aws_exam.toml
```

If you omit `--question-file`, `examexam` looks in `data/` and lets you choose from the available `.toml` files.

## What the app saves

Exam sessions are saved automatically under `.session/`.

- The session filename is based on the test name.
- Progress is written back after each answered question.
- If you restart the same exam later, `examexam` offers to resume.

That means you can do a long test across multiple sittings without losing progress.

## Terminal frontend

The default frontend is the Rich terminal UI:

```bash
examexam take --question-file data/aws_exam.toml
```

It shuffles unanswered questions, shuffles answer options, shows explanations after each answer, and keeps an interim score with timing data and confidence intervals.

## Web frontend

The web frontend runs a local FastAPI app and opens a browser:

```bash
examexam --frontend web take --question-file data/aws_exam.toml
```

Important behavior:

- It only supports the `take` command today.
- It uses the same `.session/` files as the terminal flow.
- It keeps browser-only state in memory, but the actual exam progress still lands on disk.
- It serves accessible server-rendered pages from `examexam/frontends/web_templates/`.

## Marking bad questions

In the terminal UI, after reviewing feedback, you can type `bad` instead of pressing Enter to mark the just-answered question as defective in the session file.

In the web UI, there is a dedicated "mark defective" action on the feedback page.

This does not delete the question from the original bank. It records a flag in the session data so you can review problem questions later.

## Expected question-bank shape

`take` expects a TOML file with a top-level `questions` array. Each question has text plus options, and each option must say whether it is correct.

Minimal example:

```toml
[[questions]]
question = "What is 2 + 2?"
id = "example-question-1"

[[questions.options]]
text = "4"
explanation = "Correct."
is_correct = true

[[questions.options]]
text = "5"
explanation = "Incorrect."
is_correct = false
```

For multi-select questions, mark multiple options with `is_correct = true`. `examexam` derives the answer count from the data, not from the displayed text alone.
