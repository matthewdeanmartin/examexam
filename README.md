# examexam

A CLI for **creating**, **validating**, **studying for**, and **taking** multiple-choice practice exams. Keep everything local as TOML question banks, generate new questions with an LLM, sanity-check them, generate study guides, and test your knowledge in a `rich` terminal UI.

---

## Install

### Recommended: `pipx`

```bash
pipx install examexam
````

> `pipx` keeps tools isolated and on your path. If you don’t use pipx, you can install with `python -m pip install examexam` or `uv tool install examexam`.

-----

## Quick Start: AWS Practice Exam Workflow

Here’s a typical workflow for creating and studying for an AWS exam.

1.  **Initialize a new project**

    This creates a default `examexam.toml` configuration file in your current directory.

    ```bash
    examexam init
    ```

2.  **Create a topics file**

    Make a simple text file with one topic per line. Let's call it `aws_topics.txt`:

    ```text
    AWS VPC
    AWS S3
    AWS IAM
    AWS Security Groups vs NACLs
    ```

3.  **Generate a study plan**

    Get a head start by generating a Markdown study guide for all your topics.

    ```bash
    examexam study-plan --toc-file aws_topics.txt
    ```

    > This creates a `study_guide/aws_topics_study_plan.md` file.

4.  **Generate questions**

    Create 5 questions for each topic and save them to a TOML file. We'll use a fast, cheap model for this.

    ```bash
    examexam generate \
      --toc-file aws_topics.txt \
      --output-file aws_exam.toml \
      -n 5 \
      --model-provider openai \
      --model-class fast
    ```

5.  **Validate the questions**

    Have a different LLM answer each question and flag potentially "bad" or unfair questions. The results are saved back into `aws_exam.toml`.

    ```bash
    examexam validate \
      --question-file aws_exam.toml \
      --model-provider anthropic
    ```

6.  **Drill down on a tough topic**

    Generate a detailed research guide for a specific topic you want to focus on.

    ```bash
    examexam research --topic "AWS Security Groups vs NACLs"
    ```

    > This creates a `study_guide/aws_security_groups_vs_nacls.md` file.

7.  **Take the exam**

    Launch the interactive terminal UI to take your test. Progress is saved automatically.

    ```bash
    examexam take --question-file aws_exam.toml
    ```

8.  **(Optional) Customize the prompts**

    If you want to change how questions or study guides are generated, you can deploy the default Jinja2 templates to your local directory for editing.

    ```bash
    examexam customize
    ```

    > This creates a `prompts/` folder. `examexam` will automatically use these local templates instead of the built-in ones.

-----

## Configuration

`examexam` is configured via a TOML file and environment variables.

### `examexam.toml`

Run `examexam init` to create a default `examexam.toml` file. This file allows you to set default values for command-line arguments and general behavior, such as your preferred models.

### Environment Variables

Any setting in the TOML file can be overridden by an environment variable. The format is `EXAMEXAM_SECTION_KEY`. For example, to override the default number of questions:

```bash
export EXAMEXAM_GENERAL_DEFAULT_N=10
examexam generate --toc-file ...
```

### Precedence Order

1.  Command-line arguments (e.g., `--model ...`)
2.  Environment variables (e.g., `EXAMEXAM_GENERATE_MODEL=...`)
3.  Values in `examexam.toml`
4.  Hardcoded application defaults

-----

## Model Selection

You can control which LLM is used with three flags, in order of priority:

1.  `--model`: Specifies an exact model ID (e.g., `gpt-4.1-mini`). This overrides all other settings.
2.  `--model-provider`: Chooses a provider like `openai`, `anthropic`, `google`, etc.
3.  `--model-class`: Use either `fast` (default, for cheaper, quicker models) or `frontier` (for more powerful, expensive models). Used in combination with `--model-provider`.

**Examples:**

  * `--model-provider google --model-class frontier` -\> Uses Google's best model.
  * `--model-provider openai` -\> Uses OpenAI's default "fast" model.
  * `--model gpt-5` -\> Uses exactly `gpt-5`, ignoring provider and class flags.

-----

## Command Reference

```text
# General
examexam --help
examexam --version
examexam --verbose      # Enable detailed logging for any command

# Setup and Customization
examexam init
examexam customize [--target-dir <path>] [--force]

# Core Workflow
examexam generate --toc-file <path> [-n 5] [--output-file <path>] [--model-provider ...]
examexam validate --question-file <path> [--model-provider ...]
examexam take [--question-file <path>]
examexam convert --input-file <path> --output-base-name <str>

# Study Tools
examexam research --topic <str> [--model-provider ...]
examexam study-plan --toc-file <path> [--model-provider ...]
```

### Command Details

  * **`init`**: Creates a default `examexam.toml` configuration file in the current directory.
  * **`generate`**: Creates new multiple-choice questions for topics in a `--toc-file` and appends them to a TOML question bank.
  * **`validate`**: Asks an LLM to answer each question and flags questions as "Good" or "Bad" with a rationale, saving results back to the file.
  * **`take`**: Launches a clean, keyboard-only test UI. Supports automatic session saving and resuming. Can also be run in a non-interactive "machine mode" for testing (`--machine --strategy <name>`).
  * **`convert`**: Turns a TOML question bank into pretty Markdown and HTML files for studying.
  * **`research`**: Generates a detailed, single-topic study guide with concepts, examples, and suggested search queries.
  * **`study-plan`**: Generates a consolidated Markdown study guide covering all topics from a `--toc-file`.
  * **`customize`**: Deploys the built-in Jinja2 prompt templates to a local `prompts/` directory so you can edit them.

-----

## File Locations

  * **Configuration:** `examexam.toml` in your project's root directory.
  * **Question Banks (TOML):** You decide where to store them. We recommend a `data/` folder (e.g., `data/aws_exam.toml`).
  * **Topic Files (TXT):** Plain text files with one topic per line.
  * **Session Files:** Saved automatically to `.session/<test-name>.toml` to allow resuming tests.
  * **Study Guides:** Saved to `study_guide/` in your project directory.
  * **Custom Prompts:** If you run `customize`, templates are deployed to `prompts/`.

## Credits

* **Author:** Matthew Dean Martin (matthewdeanmartin)
* Thanks to **OpenAI** and **Google Gemini** models used during generation/validation.

## License

MIT License
