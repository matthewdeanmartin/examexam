# Installation

## Recommended install

```bash
pipx install examexam
```

If you prefer, you can also use:

```bash
python -m pip install examexam
```

Or:

```bash
uv tool install examexam
```

## Web frontend

The browser frontend is optional. Install the extra dependencies if you want `--frontend web`:

```bash
pip install "examexam[web]"
```

## Developer install

For local development in this repository:

```bash
git clone https://github.com/matthewdeanmartin/examexam.git
cd examexam
uv sync --group dev
```

Useful follow-up commands:

```bash
uv run examexam --help
uv run mkdocs serve
```
