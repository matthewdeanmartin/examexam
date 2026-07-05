# examexam

`examexam` is a local-first exam toolkit for four related jobs:

- generate new multiple-choice questions from a topic list
- validate question quality with deterministic checks plus an LLM reviewer
- turn a question bank into prettier study material
- take the exam in a terminal UI or localhost web UI

The core storage format is a TOML question bank. That keeps the workflow inspectable and easy to version in git.

Typical flow:

1. Create a topics file with one topic per line.
2. Run `examexam generate` to build a question bank.
3. Run `examexam validate` to add review output back into that bank.
4. Run `examexam take` or `examexam --frontend web take` to study from it.

The repository `README.md` is still the best quick overview for badges, release links, and copy-paste examples. This docs site expands the README into audience-specific guides.
