# Web Frontend Plan

## Overview

Implement a web-based frontend for examexam using a local HTTP server. This allows
users to take exams in their browser with a modern responsive UI. The backend serves
both the API and static HTML/JS/CSS.

## Architecture Choice: FastAPI + HTMX (or plain JS)

Two viable approaches:

### Option A: FastAPI + Jinja2 server-rendered pages (recommended for simplicity)
- Each page is a server-rendered HTML template
- Form submissions via POST, redirects via PRG pattern
- Minimal JavaScript needed
- Consistent with the project's existing Jinja2 usage

### Option B: FastAPI API + Separate SPA
- FastAPI serves a JSON REST API
- Frontend is a standalone React/Vue/Svelte app
- More complex but more interactive
- Requires a build step

**Recommendation: Option A** — aligns with project philosophy (Python-focused, minimal deps).

## Files to Create

```
examexam/frontends/
  web_ui.py                  # FrontendUI implementation + FastAPI app
  web_templates/
    base.html                # Base layout (header, nav, footer)
    test_selection.html      # Test picker
    question.html            # Question display + answer form
    feedback.html            # Answer feedback
    results.html             # Score display
    progress.html            # Generation/validation progress
  web_static/
    style.css                # Styles
    app.js                   # Minimal JS (timer, auto-refresh for progress)
```

## WebUI Class Design

```python
class WebUI:
    """Web frontend implementing FrontendUI protocol.

    Unlike CLI/GUI, the web frontend is inherently request-response.
    The FrontendUI methods map to HTTP endpoints, and state is kept
    in server-side session storage (dict keyed by session ID).
    """

    def __init__(self, host="127.0.0.1", port=8080):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self._state = {}  # session state
        self._setup_routes()

    def run(self, callback=None):
        """Start the web server. callback starts the exam flow."""
        if callback:
            # Run callback in background thread when user hits "Start"
            ...
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
```

## Route Mapping

| Route | Method | Purpose |
|---|---|---|
| `GET /` | — | Test selection page |
| `POST /start` | — | Start exam, redirect to first question |
| `GET /question/{n}` | — | Display question N |
| `POST /answer/{n}` | — | Submit answer, show feedback |
| `GET /feedback/{n}` | — | Answer feedback page |
| `POST /next/{n}` | — | Advance to next question |
| `GET /results` | — | Final results page |
| `GET /progress` | — | Generation/validation progress (SSE or polling) |

## FrontendUI Protocol Adaptation

The web frontend is fundamentally different from CLI/GUI because it's stateless
and request-driven. The adaptation strategy:

1. **State machine**: Store exam state (current question, session, score) in a
   server-side dict keyed by browser session cookie.

2. **Blocking methods become endpoints**: Methods like `show_question()` + `get_answer()`
   become a GET (render question) + POST (process answer) pair.

3. **Progress uses SSE**: `progress_start/update/finish` map to Server-Sent Events
   that the browser consumes with EventSource.

4. **The `run()` method starts uvicorn**.

### Synchronization Approach

Instead of threading.Event (used by Tkinter/TUI), the web frontend uses a
request-response flow:

```
Browser                          Server
  |-- GET /question/1 ------------>|
  |<-- HTML with question ---------|
  |                                |
  |-- POST /answer/1 ------------->|  (scores answer, stores result)
  |<-- redirect to /feedback/1 ----|
  |                                |
  |-- GET /feedback/1 ------------>|
  |<-- HTML with feedback ---------|
  |                                |
  |-- POST /next/1 --------------->|
  |<-- redirect to /question/2 ----|
```

## HTML Templates

Use Jinja2 (already a dependency). Example `question.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="question-panel">
  <h2>Question {{ n }} of {{ total }}</h2>
  <div class="question-text">{{ question.question | markdown }}</div>

  <form method="POST" action="/answer/{{ n }}">
    {% for opt in options %}
    <label class="option">
      <input type="checkbox" name="answer" value="{{ loop.index }}">
      {{ loop.index }}. {{ opt.text }}
    </label>
    {% endfor %}
    <button type="submit">Submit Answer</button>
  </form>
</div>
{% endblock %}
```

## Dependencies

Add to pyproject.toml (optional dependency group):

```toml
[project.optional-dependencies]
web = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "python-multipart>=0.0.5",
]
```

Users install with: `pip install examexam[web]`

## Session Management

Use a simple in-memory dict with a cookie-based session ID:

```python
import uuid
from fastapi import Cookie, Response

sessions: dict[str, dict] = {}

@app.get("/")
def index(session_id: str = Cookie(default=None), response: Response):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id)
    ...
```

For production use, this could be swapped to Redis/database, but in-memory
is fine for the local-server use case.

## Progress Reporting (for generate/validate)

Use Server-Sent Events (SSE):

```python
from sse_starlette.sse import EventSourceResponse

@app.get("/progress-stream")
async def progress_stream():
    async def event_generator():
        while not done:
            yield {"data": json.dumps({"current": n, "total": total, "desc": desc})}
            await asyncio.sleep(0.5)
    return EventSourceResponse(event_generator())
```

Browser-side:
```javascript
const source = new EventSource("/progress-stream");
source.onmessage = (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("progress-bar").style.width = `${data.current/data.total*100}%`;
};
```

## Registration

In `examexam/frontends/__init__.py`:
```python
if name == "web":
    from examexam.frontends.web_ui import WebUI
    return WebUI()
```

## Testing

- Use FastAPI's `TestClient` for endpoint testing
- Mock the Router.call as in other tests
- Test the state machine transitions

## Estimated Effort

Large. Building a full web UI with templates, routes, session management, and SSE
progress. Estimate: ~500-800 lines of Python + ~200 lines of HTML/CSS/JS.

## Future Enhancements

- WebSocket support for real-time question flow
- Multi-user support with proper session storage
- REST API mode (JSON responses) for building custom frontends
- Mobile-responsive CSS
