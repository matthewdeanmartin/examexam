# ExamExam Multi-Frontend Architecture

## Overview

ExamExam supports multiple frontend interfaces through a **UI Protocol** abstraction layer.
Business logic never touches UI directly — it calls methods on a `FrontendUI` protocol object,
and each frontend (CLI/Rich, Tkinter GUI, Textual TUI, Web) provides its own implementation.

## Architecture Diagram

```
                    +-------------------+
                    |   __main__.py     |
                    | (arg parsing,     |
                    |  frontend select) |
                    +--------+----------+
                             |
                    selects frontend
                             |
              +--------------+--------------+
              |              |              |
         +----v----+   +----v----+   +----v----+
         | RichUI  |   |TkinterUI|   | (future)|
         | (CLI)   |   | (GUI)   |   | TUI/Web |
         +---------+   +---------+   +---------+
              |              |              |
              +------+-------+------+-------+
                     |              |
              implements      implements
                     |              |
              +------v--------------v------+
              |       FrontendUI           |
              |       (Protocol)           |
              +------------+---------------+
                           |
                    called by
                           |
              +------------v---------------+
              |     Business Logic         |
              | take_exam, generate_qs,    |
              | validate, research, etc.   |
              +----------------------------+
              |     Core Models            |
              | Question, Option, Session  |
              +----------------------------+
              |     APIs Layer             |
              | Router, Conversation, LLMs |
              +----------------------------+
```

## Key Design Decisions

### 1. Protocol-based UI abstraction
We use `typing.Protocol` (structural subtyping) rather than ABC. This means frontend
implementations don't need to inherit from anything — they just need to have the right methods.

### 2. FrontendUI is split into capability groups
Not every command needs every UI method. The protocol has method groups:
- **Core**: `show_message()`, `show_error()`, `confirm()`
- **Exam**: `show_question()`, `get_answer()`, `show_answer_feedback()`, `show_results()`
- **Progress**: `create_progress()`, `update_progress()`, `complete_progress()`
- **Display**: `show_panel()`, `show_table()`, `show_markdown()`

### 3. Business logic accepts `ui: FrontendUI` parameter
All `*_now()` entry functions accept a `ui` parameter. `__main__.py` instantiates the
appropriate frontend and passes it down.

### 4. Data models replace raw dicts
`examexam/models.py` defines dataclasses for `Question`, `Option`, `ExamSession`, `ExamResult`.
These provide type safety and make the protocol methods well-typed.

## Frontend Implementations

| Frontend | Module | Status |
|----------|--------|--------|
| CLI (Rich) | `examexam/frontends/rich_ui.py` | Implemented |
| GUI (Tkinter) | `examexam/frontends/tkinter_ui.py` | Implemented |
| TUI (Textual) | `examexam/frontends/textual_ui.py` | Planned |
| Web (Flask/FastAPI) | `examexam/frontends/web_ui.py` | Planned |

## File Structure

```
examexam/
  models.py              # Core data models (Question, Option, etc.)
  ui_protocol.py         # FrontendUI protocol definition
  frontends/
    __init__.py          # Frontend registry & factory
    rich_ui.py           # CLI frontend (Rich)
    tkinter_ui.py        # GUI frontend (Tkinter)
    textual_ui.py        # TUI frontend (Textual) - future
    web_ui.py            # Web frontend - future
```

## Frontend Selection

```
examexam --frontend gui take          # Tkinter GUI
examexam --frontend cli take          # Rich CLI (default)
examexam --frontend tui take          # Textual TUI (future)
examexam --frontend web take          # Web server (future)
```
