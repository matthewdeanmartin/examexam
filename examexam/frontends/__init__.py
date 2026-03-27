"""Frontend registry for examexam.

Use get_frontend() to instantiate the appropriate UI backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examexam.ui_protocol import FrontendUI

FRONTEND_CHOICES = ("cli", "gui", "tui", "web")


def get_frontend(name: str = "cli") -> FrontendUI:
    """Instantiate and return the requested frontend.

    Args:
        name: One of 'cli', 'gui', 'tui', 'web'.

    Returns:
        An object implementing the FrontendUI protocol.
    """
    if name == "cli":
        from examexam.frontends.rich_ui import RichUI

        return RichUI()
    if name == "gui":
        from examexam.frontends.tkinter_ui import TkinterUI

        return TkinterUI()
    if name == "tui":
        raise NotImplementedError("Textual TUI frontend is not yet implemented. See spec/tui_frontend_plan.md")
    if name == "web":
        try:
            from examexam.frontends.web_ui import WebUI
        except ImportError as exc:
            msg = "Web frontend dependencies are not installed. Install with: pip install examexam[web]"
            raise ImportError(msg) from exc

        return WebUI()
    raise ValueError(f"Unknown frontend: {name!r}. Choose from: {FRONTEND_CHOICES}")
