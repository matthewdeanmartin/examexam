# TUI Frontend Plan (Textual)

## Overview

Implement a Textual-based TUI frontend for examexam. Textual is already a dependency
in pyproject.toml (`textual>=1.0.0`). This frontend provides a rich terminal UI with
mouse support, scrolling panels, styled widgets, and reactive state — a significant
upgrade over the basic Rich CLI while still running in the terminal.

## File to Create

`examexam/frontends/textual_ui.py`

## Architecture

The `TextualUI` class implements the `FrontendUI` protocol from `examexam/ui_protocol.py`.

Because Textual has its own async event loop (`App.run()`), the design mirrors
the Tkinter frontend: business logic runs in a background thread and communicates
with the Textual UI thread via `threading.Event` objects for synchronization.

```python
class ExamExamApp(textual.app.App):
    """The Textual application."""
    ...

class TextualUI:
    """FrontendUI protocol implementation wrapping ExamExamApp."""
    def __init__(self):
        self._app = ExamExamApp()
        self._answer_event = threading.Event()
        ...
```

## Key Widgets to Build

### 1. TestSelectionScreen
- List of available tests with highlight navigation
- Enter to select, styled header

### 2. QuestionScreen
- Panel with the question text (Markdown rendered)
- Options displayed as selectable list items or checkboxes
- Footer with "Submit" button/keybinding
- Status bar showing question N/M, current score

### 3. FeedbackScreen
- Green/red banner for correct/incorrect
- Explanations list with color coding
- "Next" and "Mark as Bad" buttons in footer

### 4. ResultsScreen
- Score display with bar chart or gauge widget
- Time statistics
- Confidence intervals
- Pass/fail status with color

### 5. ProgressScreen (for generate/validate)
- Textual's built-in ProgressBar widget
- Real-time log scrolling below

## Protocol Method Mapping

| FrontendUI method | Textual implementation |
|---|---|
| `show_test_selection()` | Push `TestSelectionScreen`, wait for selection event |
| `show_question()` | Push `QuestionScreen` with reactive data |
| `get_answer()` | Wait on `_answer_event`, validate, loop if invalid |
| `show_answer_feedback()` | Push `FeedbackScreen` |
| `show_results()` | Push `ResultsScreen` |
| `wait_for_continue()` | Wait for key press event from `FeedbackScreen` |
| `show_message()` | Append to log panel |
| `show_error()` | Show notification/toast |
| `confirm()` | Show modal dialog, wait on event |
| `show_markdown()` | Use `Markdown` widget |
| `progress_start/update/finish()` | Manage `ProgressBar` widget |
| `run()` | `self._app.run()` |
| `shutdown()` | `self._app.exit()` |

## CSS Styling

Textual uses CSS for styling. Create `examexam/frontends/textual_ui.tcss`:

```css
QuestionScreen {
    layout: vertical;
}
QuestionScreen .question-text {
    padding: 1 2;
    background: $surface;
    border: solid $primary;
}
QuestionScreen .option {
    padding: 0 2;
}
.correct { color: green; }
.incorrect { color: red; }
ResultsScreen .score { text-style: bold; content-align: center middle; }
```

## Threading Model

```
Main thread:  TextualUI.run() → app.run() (Textual event loop)
                                    ↑
                                messages via app.call_from_thread()
                                    |
Worker thread: take_exam_now() → ui.show_question() → posts message → waits on Event
```

Use `app.call_from_thread()` for all UI updates from the business logic thread.

## Keybindings

- `Enter` / `Space`: Select/toggle option
- `s`: Submit answer
- `n`: Next question (from feedback screen)
- `b`: Mark as bad question
- `q`: Quit (with confirmation)

## Registration

In `examexam/frontends/__init__.py`, the `"tui"` branch should:
```python
if name == "tui":
    from examexam.frontends.textual_ui import TextualUI
    return TextualUI()
```

## Testing

- Mock the Textual app's `run()` to avoid starting the event loop
- Test each screen widget independently using Textual's `pilot` test framework
- Reuse the mock UI pattern from existing tests as reference

## Estimated Effort

Medium-large. The main complexity is the threading synchronization and building
5 custom screen widgets. Textual's built-in widgets (Markdown, ProgressBar, ListView)
cover most needs. Estimate: ~400-600 lines of Python + ~50 lines of CSS.
