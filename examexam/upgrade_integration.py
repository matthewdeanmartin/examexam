"""Helpers for embedding do_i_need_to_upgrade into examexam.

This replaces the old bespoke ``examexam.utils.update_checker`` module. The
``do_i_need_to_upgrade`` package owns the PyPI polling, cache, and rendering; this
module just wires it into examexam's argparse-based CLI lifecycle.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any, Literal, Protocol


class Report(Protocol):
    """Minimal report surface used by examexam."""

    is_empty: bool

    def render_text(self, *, stream: Any) -> str: ...


class Settings(Protocol):
    """Minimal settings surface used by examexam."""

    def replace(self, *, allow_network: bool, notify: str) -> Settings: ...


class SettingsFactory(Protocol):
    """Callable protocol for constructing settings instances."""

    def __call__(self, *, dist_name: str, position: str, notify: str) -> Settings: ...


class AddCommand(Protocol):
    """Callable protocol for registering an upgrade-related subcommand."""

    def __call__(
        self,
        subparsers: argparse._SubParsersAction[Any],
        dist_name: str,
        *,
        command: str = ...,
        settings: Settings | None = ...,
    ) -> None: ...


class RunUpgradeCommand(Protocol):
    """Callable protocol for dispatching an upgrade-related CLI command."""

    def __call__(self, args: argparse.Namespace) -> int | None: ...


class CheckForUpdates(Protocol):
    """Callable protocol for reading or refreshing the upgrade report."""

    def __call__(
        self,
        host: Any | None = ...,
        position: Literal["start", "end", "both", "off"] = ...,
        background: bool = ...,
        force: bool = ...,
        spawn: bool = ...,
        settings: Settings | None = ...,
    ) -> Report: ...


add_check_command: AddCommand | None
add_upgrade_command: AddCommand | None
run_if_upgrade_command: RunUpgradeCommand | None
check_for_updates: CheckForUpdates | None
SettingsClass: SettingsFactory | None

try:
    upgrade_module = importlib.import_module("do_i_need_to_upgrade")
    upgrade_api_module = importlib.import_module("do_i_need_to_upgrade.api")
    upgrade_settings_module = importlib.import_module("do_i_need_to_upgrade.settings")

    add_check_command = upgrade_module.add_check_command
    add_upgrade_command = upgrade_module.add_upgrade_command
    run_if_upgrade_command = upgrade_module.run_if_upgrade_command
    check_for_updates = upgrade_api_module.check_for_updates
    SettingsClass = upgrade_settings_module.Settings
    HAS_UPGRADE_SUPPORT = True
except ImportError:
    add_check_command = None
    add_upgrade_command = None
    run_if_upgrade_command = None
    check_for_updates = None
    SettingsClass = None
    HAS_UPGRADE_SUPPORT = False

DIST_NAME = "examexam"
CHECK_UPDATES_COMMAND = "check-updates"
UPGRADE_COMMAND = "upgrade"
UPGRADE_COMMANDS = frozenset({CHECK_UPDATES_COMMAND, UPGRADE_COMMAND})


def settings() -> Settings:
    """Return examexam's embedded update-check settings."""
    assert SettingsClass is not None
    return SettingsClass(dist_name=DIST_NAME, position="start", notify="return-only")


def should_handle_upgrade_command(argv: Sequence[str]) -> bool:
    """Return True when argv selects an integrated update subcommand."""
    if not HAS_UPGRADE_SUPPORT:
        return False
    for token in argv:
        if token == "--":  # nosec
            return False
        if token.startswith("-"):
            continue
        return token in UPGRADE_COMMANDS
    return False


def add_commands(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register integrated do_i_need_to_upgrade subcommands."""
    if not HAS_UPGRADE_SUPPORT:
        return
    active_settings = settings()
    assert add_upgrade_command is not None
    assert add_check_command is not None
    add_upgrade_command(subparsers, DIST_NAME, command=UPGRADE_COMMAND, settings=active_settings)
    add_check_command(subparsers, DIST_NAME, command=CHECK_UPDATES_COMMAND, settings=active_settings)


def startup_report() -> Report | None:
    """Kick off the background refresh and return the current cache-backed report."""
    if not HAS_UPGRADE_SUPPORT:
        return None
    assert check_for_updates is not None
    report = check_for_updates(settings=settings())
    return report if report.is_empty is False else None


def exit_report() -> Report | None:
    """Read the refreshed cache on exit without doing more network I/O."""
    if not HAS_UPGRADE_SUPPORT:
        return None
    assert check_for_updates is not None
    report = check_for_updates(settings=settings().replace(allow_network=False, notify="return-only"))
    return report if report.is_empty is False else None


def render_notice(report: Report | None) -> str:
    """Render a user-facing update notice for stderr."""
    if report is None:
        return ""
    return report.render_text(stream=sys.stderr)


def run_command(args: argparse.Namespace) -> int:
    """Dispatch an already-parsed integrated update-related subcommand."""
    if not HAS_UPGRADE_SUPPORT:
        return 0
    assert run_if_upgrade_command is not None
    result = run_if_upgrade_command(args)
    return 0 if result is None else result


__all__ = [
    "CHECK_UPDATES_COMMAND",
    "DIST_NAME",
    "HAS_UPGRADE_SUPPORT",
    "UPGRADE_COMMAND",
    "UPGRADE_COMMANDS",
    "add_commands",
    "exit_report",
    "render_notice",
    "run_command",
    "settings",
    "should_handle_upgrade_command",
    "startup_report",
]
