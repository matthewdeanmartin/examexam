"""FastAPI-based web frontend for taking exams in a browser."""

from __future__ import annotations

import copy
import random
import re
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Thread, Timer
from typing import Any

import rtoml as toml
from fastapi import Cookie, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown import markdown
from markupsafe import Markup

from examexam.models import AnswerFeedback, ExamResult, Option, ProgressInfo, Question, TestInfo
from examexam.take_exam import (
    _build_exam_result,
    build_answer_feedback,
    find_select_pattern,
    get_session_path,
    humanize_timedelta,
    is_valid,
    load_questions,
    save_session_file,
)

COOKIE_NAME = "examexam_session_id"


@dataclass
class ResumeInfo:
    """Summary of a resumable session for a test."""

    completed: int
    total: int
    time_ago: str


@dataclass
class AvailableTest:
    """A test the browser UI can launch."""

    name: str
    question_file: Path
    resume_info: ResumeInfo | None = None


@dataclass
class BrowserSession:
    """Per-browser in-memory state."""

    session_id: str
    test_name: str | None = None
    question_file: Path | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    session: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime | None = None
    question_order: list[str] = field(default_factory=list)
    option_orders: dict[str, list[int]] = field(default_factory=dict)
    flash_message: str | None = None
    flash_level: str = "info"

    @property
    def has_active_exam(self) -> bool:
        return bool(self.test_name and self.question_file and self.start_time and self.question_order)

    def completed_count(self) -> int:
        return sum(1 for question in self.session if question.get("user_score") is not None)

    def score(self) -> int:
        return sum(1 for question in self.session if question.get("user_score") == 1)

    def clear_exam(self) -> None:
        self.test_name = None
        self.question_file = None
        self.questions = []
        self.session = []
        self.start_time = None
        self.question_order = []
        self.option_orders = {}


class WebUI:
    """Accessible localhost web UI using FastAPI and server-rendered templates."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._configured_question_file: Path | None = None
        self._browser_sessions: dict[str, BrowserSession] = {}
        self._progress_items: dict[str, ProgressInfo] = {}

        templates_dir = Path(__file__).with_name("web_templates")
        static_dir = Path(__file__).with_name("web_static")

        self.app = FastAPI(title="Examexam Web UI")
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.templates = Jinja2Templates(directory=str(templates_dir))
        self.templates.env.filters["markdown"] = self._render_markdown
        self.templates.env.filters["humanize_timedelta"] = humanize_timedelta

        self._setup_routes()

    # ---- Core ----

    def configure_take_exam(self, question_file: str | None) -> None:
        """Set the optional exam file to launch from the browser UI."""
        self._configured_question_file = Path(question_file).resolve() if question_file else None

    def show_message(self, message: str, *, style: str = "") -> None:
        level = style or "info"
        for session in self._browser_sessions.values():
            session.flash_message = message
            session.flash_level = level

    def show_error(self, message: str) -> None:
        self.show_message(message, style="error")

    def confirm(self, message: str, *, default: bool = True) -> bool:
        self.show_message(message, style="warning")
        return default

    def get_input(self, prompt: str) -> str:
        raise RuntimeError(f"Web frontend cannot synchronously answer prompt: {prompt}")

    # ---- Exam taking ----

    def show_test_selection(self, tests: list[TestInfo]) -> int | None:
        return tests[0].index - 1 if tests else None

    def show_session_info(self, test_name: str, completed: int, total: int, time_ago: str) -> None:
        self.show_message(
            f"Found a session for {test_name}: {completed}/{total} complete, started {time_ago} ago.",
            style="info",
        )

    def show_question(self, question: Question, options: list[Option], question_number: int | None = None) -> None:
        _ = (question, options, question_number)

    def get_answer(self, option_count: int, answer_count: int) -> str:
        raise RuntimeError(
            f"Web frontend cannot synchronously collect {answer_count} answers from {option_count} options."
        )

    def show_answer_feedback(self, feedback: AnswerFeedback) -> None:
        self.show_message("Answer submitted.", style="success" if feedback.is_correct else "warning")

    def show_results(self, result: ExamResult) -> None:
        label = "Passed" if result.passed else "Completed"
        self.show_message(f"{label}: {result.score}/{result.total} ({result.percent:.1f}%)", style="info")

    def wait_for_continue(self) -> str:
        return ""

    def clear_screen(self) -> None:
        return None

    # ---- Progress ----

    def progress_start(self, total: int, description: str = "") -> str:
        task_id = str(uuid.uuid4())
        self._progress_items[task_id] = ProgressInfo(task_id=task_id, description=description, total=total, current=0)
        return task_id

    def progress_update(self, task_id: str, advance: int = 1, description: str = "") -> None:
        progress = self._progress_items.get(task_id)
        if progress is None:
            return
        progress.current = min(progress.total, progress.current + advance)
        if description:
            progress.description = description

    def progress_finish(self, task_id: str) -> None:
        progress = self._progress_items.get(task_id)
        if progress is not None:
            progress.current = progress.total

    # ---- Display ----

    def show_panel(self, content: str, *, title: str = "", style: str = "") -> None:
        header = f"{title}: " if title else ""
        self.show_message(f"{header}{content}", style=style or "info")

    def show_markdown(self, content: str) -> None:
        self.show_message(content, style="markdown")

    def show_rule(self, title: str = "") -> None:
        self.show_message(title, style="rule")

    # ---- Lifecycle ----

    def run(self, callback: Any = None) -> None:
        if callback is not None:
            Thread(target=callback, daemon=True).start()
        Timer(0.75, lambda: webbrowser.open(self.root_url)).start()
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    def shutdown(self) -> None:
        self._browser_sessions.clear()
        self._progress_items.clear()

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def _setup_routes(self) -> None:
        @self.app.get("/")
        def index(
            request: Request,
            response: Response,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            page = self._render(
                "test_selection.html",
                request,
                browser_session,
                available_tests=self._get_available_tests(),
                active_exam_url=self._active_exam_url(browser_session),
            )
            if created_session_id:
                page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return page

        @self.app.post("/start")
        async def start_exam(
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            form = await request.form()
            requested_test_name = str(form.get("test_name", "")).strip()
            resume_requested = str(form.get("resume", "false")).lower() == "true"

            selected_test = self._resolve_selected_test(requested_test_name)
            if selected_test is None:
                browser_session.flash_message = "Select a valid exam file to continue."
                browser_session.flash_level = "error"
                response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            start_time = datetime.now()
            if resume_requested and selected_test.resume_info is not None:
                session_data, saved_start_time = self._load_saved_session(selected_test.name)
                session = session_data
                start_time = saved_start_time or start_time
            else:
                session = copy.deepcopy(load_questions(str(selected_test.question_file)))
                session_path = get_session_path(selected_test.name)
                if session_path.exists():
                    session_path.unlink()

            questions = load_questions(str(selected_test.question_file))
            if not session:
                session = copy.deepcopy(questions)

            question_order = self._build_question_order(questions, session)
            browser_session.test_name = selected_test.name
            browser_session.question_file = selected_test.question_file
            browser_session.questions = questions
            browser_session.session = session
            browser_session.start_time = start_time
            browser_session.question_order = question_order
            browser_session.option_orders = {}

            save_session_file(get_session_path(selected_test.name), browser_session.session, start_time)

            destination = (
                "/results"
                if browser_session.completed_count() >= len(browser_session.question_order)
                else f"/question/{browser_session.completed_count() + 1}"
            )
            response = RedirectResponse(url=destination, status_code=status.HTTP_303_SEE_OTHER)
            if created_session_id:
                response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return response

        @self.app.get("/question/{n}")
        def question_page(
            n: int,
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            redirect_target = self._validate_question_route(browser_session, n)
            if redirect_target is not None:
                response = RedirectResponse(url=redirect_target, status_code=status.HTTP_303_SEE_OTHER)
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            question, session_question = self._question_pair_for_number(browser_session, n)
            if session_question.get("start_time") is None and browser_session.start_time is not None:
                session_question["start_time"] = datetime.now().isoformat()
                save_session_file(
                    get_session_path(browser_session.test_name or "session"),
                    browser_session.session,
                    browser_session.start_time,
                )

            page = self._render_question(
                request,
                browser_session,
                question_number=n,
                question=question,
                selected_answers=[],
                error_message=None,
            )
            if created_session_id:
                page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return page

        @self.app.post("/answer/{n}")
        async def submit_answer(
            n: int,
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            redirect_target = self._validate_question_route(browser_session, n)
            if redirect_target is not None:
                response = RedirectResponse(url=redirect_target, status_code=status.HTTP_303_SEE_OTHER)
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            question, session_question = self._question_pair_for_number(browser_session, n)
            form = await request.form()
            submitted_answers = [value for value in form.getlist("answer") if isinstance(value, str)]
            selected_answers = submitted_answers
            display_options = self._display_options(browser_session, question)
            question_model = Question.from_dict(question)

            answer_text = ",".join(selected_answers)
            valid, error_message = is_valid(
                answer_text,
                option_count=len(display_options),
                answer_count=question_model.answer_count,
                last_is_bad_question_flag=False,
            )
            if not valid:
                page = self._render_question(
                    request,
                    browser_session,
                    question_number=n,
                    question=question,
                    selected_answers=selected_answers,
                    error_message=error_message,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
                if created_session_id:
                    page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return page

            selected = [display_options[int(answer) - 1] for answer in selected_answers]
            feedback = build_answer_feedback(display_options, selected)
            if session_question.get("start_time") is None:
                session_question["start_time"] = datetime.now().isoformat()
            session_question["completion_time"] = datetime.now().isoformat()
            session_question["user_answers"] = sorted(feedback.user_answers)
            session_question["user_score"] = 1 if feedback.is_correct else 0
            save_session_file(
                get_session_path(browser_session.test_name or "session"),
                browser_session.session,
                browser_session.start_time or datetime.now(),
            )

            response = RedirectResponse(url=f"/feedback/{n}", status_code=status.HTTP_303_SEE_OTHER)
            if created_session_id:
                response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return response

        @self.app.get("/feedback/{n}")
        def feedback_page(
            n: int,
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            if not browser_session.has_active_exam:
                response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            question, session_question = self._question_pair_for_number(browser_session, n)
            if session_question.get("user_score") is None:
                response = RedirectResponse(
                    url=f"/question/{browser_session.completed_count() + 1}", status_code=status.HTTP_303_SEE_OTHER
                )
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            display_options = self._display_options(browser_session, question)
            selected = self._selected_options(display_options, session_question.get("user_answers", []))
            feedback = build_answer_feedback(display_options, selected)
            page = self._render(
                "feedback.html",
                request,
                browser_session,
                question_number=n,
                total_questions=len(browser_session.question_order),
                question=self._question_context(question),
                feedback=feedback,
                option_details=self._feedback_option_details(display_options, session_question),
                next_url=(
                    "/results"
                    if browser_session.completed_count() >= len(browser_session.question_order)
                    else f"/question/{n + 1}"
                ),
                next_label=(
                    "View results"
                    if browser_session.completed_count() >= len(browser_session.question_order)
                    else "Next question"
                ),
            )
            if created_session_id:
                page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return page

        @self.app.post("/next/{n}")
        def next_question(
            n: int,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            if not browser_session.has_active_exam:
                response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
            elif browser_session.completed_count() >= len(browser_session.question_order):
                response = RedirectResponse(url="/results", status_code=status.HTTP_303_SEE_OTHER)
            else:
                response = RedirectResponse(
                    url=f"/question/{min(browser_session.completed_count() + 1, len(browser_session.question_order))}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            if created_session_id:
                response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return response

        @self.app.post("/mark-defective/{n}")
        def mark_defective(
            n: int,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            if browser_session.has_active_exam:
                _, session_question = self._question_pair_for_number(browser_session, n)
                session_question["defective"] = True
                save_session_file(
                    get_session_path(browser_session.test_name or "session"),
                    browser_session.session,
                    browser_session.start_time or datetime.now(),
                )
                browser_session.flash_message = "This question has been marked as defective in the session file."
                browser_session.flash_level = "warning"
                response = RedirectResponse(url=f"/feedback/{n}", status_code=status.HTTP_303_SEE_OTHER)
            else:
                response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
            if created_session_id:
                response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return response

        @self.app.get("/results")
        def results_page(
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            if not browser_session.has_active_exam:
                response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
                if created_session_id:
                    response.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
                return response

            if browser_session.start_time is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing exam start time."
                )

            result = _build_exam_result(
                browser_session.score(),
                len(browser_session.question_order),
                browser_session.start_time,
                browser_session.session,
                is_final=True,
            )
            page = self._render(
                "results.html",
                request,
                browser_session,
                result=result,
                total_questions=len(browser_session.question_order),
                session_path=str(get_session_path(browser_session.test_name or "session")),
            )
            if created_session_id:
                page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return page

        @self.app.get("/progress")
        def progress_page(
            request: Request,
            session_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
        ) -> Response:
            browser_session, created_session_id = self._get_browser_session(session_id)
            page = self._render(
                "progress.html",
                request,
                browser_session,
                progress_items=sorted(self._progress_items.values(), key=lambda item: item.description),
            )
            if created_session_id:
                page.set_cookie(COOKIE_NAME, created_session_id, httponly=True, samesite="lax")
            return page

    def _render(
        self,
        template_name: str,
        request: Request,
        browser_session: BrowserSession,
        *,
        status_code: int = status.HTTP_200_OK,
        **context: Any,
    ) -> Response:
        flash = self._pop_flash(browser_session)
        return self.templates.TemplateResponse(
            request,
            template_name,
            {
                "page_title": "Examexam web UI",
                "flash": flash,
                "active_exam_url": self._active_exam_url(browser_session),
                "active_exam_name": browser_session.test_name,
                **context,
            },
            status_code=status_code,
        )

    def _render_question(
        self,
        request: Request,
        browser_session: BrowserSession,
        *,
        question_number: int,
        question: dict[str, Any],
        selected_answers: list[str],
        error_message: str | None,
        status_code: int = status.HTTP_200_OK,
    ) -> Response:
        display_options = self._display_options(browser_session, question)
        question_model = Question.from_dict(question)
        return self._render(
            "question.html",
            request,
            browser_session,
            status_code=status_code,
            question_number=question_number,
            total_questions=len(browser_session.question_order),
            question=self._question_context(question),
            input_type="radio" if question_model.answer_count == 1 else "checkbox",
            answer_count=question_model.answer_count,
            options=self._option_context(display_options),
            selected_answers=selected_answers,
            error_message=error_message,
        )

    def _render_markdown(self, value: str) -> Markup:
        return Markup(markdown(value, extensions=["extra", "nl2br"]))

    def _get_browser_session(self, session_id: str | None) -> tuple[BrowserSession, str | None]:
        if session_id and session_id in self._browser_sessions:
            return self._browser_sessions[session_id], None
        new_session_id = str(uuid.uuid4())
        browser_session = BrowserSession(session_id=new_session_id)
        self._browser_sessions[new_session_id] = browser_session
        return browser_session, new_session_id

    def _get_available_tests(self) -> list[AvailableTest]:
        if self._configured_question_file is not None:
            if not self._configured_question_file.exists():
                return []
            name = self._configured_question_file.stem
            return [
                AvailableTest(
                    name=name,
                    question_file=self._configured_question_file,
                    resume_info=self._read_resume_info(name),
                )
            ]

        data_dir = Path("data")
        if not data_dir.exists():
            return []

        tests = []
        for test_file in sorted(data_dir.glob("*.toml")):
            tests.append(
                AvailableTest(
                    name=test_file.stem,
                    question_file=test_file.resolve(),
                    resume_info=self._read_resume_info(test_file.stem),
                )
            )
        return tests

    def _resolve_selected_test(self, requested_test_name: str) -> AvailableTest | None:
        available_tests = self._get_available_tests()
        if self._configured_question_file is not None:
            return available_tests[0] if available_tests else None
        for available_test in available_tests:
            if available_test.name == requested_test_name:
                return available_test
        return None

    def _read_resume_info(self, test_name: str) -> ResumeInfo | None:
        session_path = get_session_path(test_name)
        if not session_path.exists():
            return None
        data = toml.load(session_path)
        session_questions = data.get("questions", [])
        completed = sum(1 for question in session_questions if question.get("user_score") is not None)
        total = len(session_questions)
        if completed == 0 or total == 0:
            return None

        start_time_text = data.get("start_time")
        time_ago = "unknown"
        if isinstance(start_time_text, str):
            try:
                start_time = datetime.fromisoformat(start_time_text)
            except ValueError:
                start_time = None
            if start_time is not None:
                time_ago = humanize_timedelta(datetime.now() - start_time)
        return ResumeInfo(completed=completed, total=total, time_ago=time_ago)

    def _load_saved_session(self, test_name: str) -> tuple[list[dict[str, Any]], datetime | None]:
        data = toml.load(get_session_path(test_name))
        start_time = data.get("start_time")
        parsed_start = datetime.fromisoformat(start_time) if isinstance(start_time, str) else None
        return list(data.get("questions", [])), parsed_start

    def _build_question_order(self, questions: list[dict[str, Any]], session: list[dict[str, Any]]) -> list[str]:
        session_by_id = {question.get("id", ""): question for question in session}
        completed_ids: list[str] = []
        unanswered_ids: list[str] = []
        for question in questions:
            question_id = str(question.get("id", ""))
            if session_by_id.get(question_id, {}).get("user_score") is None:
                unanswered_ids.append(question_id)
            else:
                completed_ids.append(question_id)
        random.shuffle(unanswered_ids)
        return completed_ids + unanswered_ids

    def _active_exam_url(self, browser_session: BrowserSession) -> str | None:
        if not browser_session.has_active_exam:
            return None
        if browser_session.completed_count() >= len(browser_session.question_order):
            return "/results"
        return f"/question/{browser_session.completed_count() + 1}"

    def _pop_flash(self, browser_session: BrowserSession) -> dict[str, str] | None:
        if browser_session.flash_message is None:
            return None
        payload = {"message": browser_session.flash_message, "level": browser_session.flash_level}
        browser_session.flash_message = None
        browser_session.flash_level = "info"
        return payload

    def _validate_question_route(self, browser_session: BrowserSession, n: int) -> str | None:
        if not browser_session.has_active_exam:
            return "/"
        completed = browser_session.completed_count()
        if completed >= len(browser_session.question_order):
            return "/results"
        expected_question = completed + 1
        if n != expected_question:
            return f"/question/{expected_question}"
        return None

    def _question_pair_for_number(
        self, browser_session: BrowserSession, n: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if n < 1 or n > len(browser_session.question_order):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")
        question_id = browser_session.question_order[n - 1]
        question_map = {question.get("id", ""): question for question in browser_session.questions}
        session_map = {question.get("id", ""): question for question in browser_session.session}
        question = question_map.get(question_id)
        session_question = session_map.get(question_id)
        if question is None or session_question is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")
        return question, session_question

    def _display_options(self, browser_session: BrowserSession, question: dict[str, Any]) -> list[dict[str, Any]]:
        question_id = str(question.get("id", ""))
        option_order = browser_session.option_orders.get(question_id)
        if option_order is None:
            option_order = list(range(len(question.get("options", []))))
            random.shuffle(option_order)
            browser_session.option_orders[question_id] = option_order
        return [question["options"][index] for index in option_order]

    def _selected_options(
        self, display_options: list[dict[str, Any]], selected_answers: list[str]
    ) -> list[dict[str, Any]]:
        selected_answer_set = set(selected_answers)
        return [option for option in display_options if option.get("text") in selected_answer_set]

    def _question_context(self, question: dict[str, Any]) -> dict[str, Any]:
        question_model = Question.from_dict(question)
        return {
            "id": question_model.id,
            "question": self._format_question_text(question_model.question, question_model.answer_count),
        }

    def _option_context(self, display_options: list[dict[str, Any]]) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for index, option in enumerate(display_options, start=1):
            options.append(
                {"index": index, "text": option.get("text", ""), "explanation": option.get("explanation", "")}
            )
        return options

    def _feedback_option_details(
        self, display_options: list[dict[str, Any]], session_question: dict[str, Any]
    ) -> list[dict[str, Any]]:
        selected_answers = set(session_question.get("user_answers", []))
        details: list[dict[str, Any]] = []
        for index, option in enumerate(display_options, start=1):
            is_correct = bool(option.get("is_correct", False))
            was_selected = option.get("text", "") in selected_answers
            details.append(
                {
                    "index": index,
                    "text": option.get("text", ""),
                    "explanation": option.get("explanation", ""),
                    "is_correct": is_correct,
                    "was_selected": was_selected,
                }
            )
        return details

    def _format_question_text(self, question_text: str, answer_count: int) -> str:
        select_pattern = find_select_pattern(question_text)
        correct_select = f"(Select {answer_count})"
        if select_pattern:
            return question_text.replace(select_pattern, correct_select)
        if "(Select n)" in question_text:
            return question_text.replace("(Select n)", correct_select)
        if re.search(r"\(Select\s+\d+\)", question_text):
            return re.sub(r"\(Select\s+\d+\)", correct_select, question_text, count=1)
        return f"{question_text} {correct_select}"
