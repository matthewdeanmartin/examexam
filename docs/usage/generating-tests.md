# Creating Tests

This page is for people building question banks, not just taking them.

## Start with a topics file

Create a plain text file with one topic per line:

```text
AWS VPC
AWS IAM
AWS S3
Security Groups vs NACLs
```

## Generate questions

```bash
examexam generate \
  --toc-file aws_topics.txt \
  --output-file data/aws_exam.toml \
  -n 5 \
  --model-provider openai \
  --model-class fast
```

What happens under the hood:

- `generate_questions.py` renders `examexam/prompts/generate.md.j2`
- the prompt is sent through the shared LLM router
- the response must contain TOML matching the expected schema
- new questions are appended to the output file with generated UUIDs

If a topic fails validation or parsing, the generator retries with a corrective prompt.

## Validate questions

Generation only creates candidates. Validation is the pass that helps you spot weak or malformed questions.

```bash
examexam validate \
  --question-file data/aws_exam.toml \
  --model-provider anthropic
```

Validation does two kinds of work:

1. Deterministic checks.
2. LLM review.

Deterministic checks currently look for issues like:

- no correct answer
- duplicate option text
- duplicate question text
- banned option patterns such as "all of the above"
- mismatch between "select N" wording and the actual number of correct options

The LLM pass then:

- answers the question
- compares its answer to the marked correct answers
- records a `good` or `bad` verdict plus rationale

The output is written back into the same TOML file so you can inspect or hand-edit it.

## Generate study material

Two related commands help you study before or after creating a bank:

```bash
examexam study-plan --toc-file aws_topics.txt
examexam research --topic "Security Groups vs NACLs"
```

- `study-plan` creates a broader markdown guide in `study_guide/`
- `research` creates a topic-specific markdown guide in `study_guide/`

## Convert a bank into prettier output

```bash
examexam convert \
  --input-file data/aws_exam.toml \
  --output-base-name aws_exam
```

That produces:

- `aws_exam.md`
- `aws_exam.html`

## Pick models deliberately

You can control the provider in three ways:

- `--model` picks an exact model id
- `--model-provider` chooses the vendor
- `--model-class` picks between `fast` and `frontier`

The router resolves provider defaults in `examexam/apis/conversation_and_router.py`.

## Customize prompts

If you want to change the generation or validation prompts:

```bash
examexam customize
```

That deploys the built-in templates into `./prompts/`. After that, the local files override the bundled ones.

The deployment logic keeps hashes so rerunning `customize` can avoid overwriting templates you already edited.
