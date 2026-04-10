# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import inspect
import logging
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..data.types import LogLevel

VALID_LOG_LEVELS: tuple[LogLevel, ...] = (
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
)

LEVEL_STYLES: dict[LogLevel, str] = {
    "TRACE": "dim",
    "DEBUG": "dim",
    "INFO": "blue",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}

DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY/MM/DD HH:mm:ss}</green> | "
    "{level.icon} - <level>{message}</level>"
)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console_width: int | None = Field(default=None, ge=20)
    console_height: int | None = Field(default=None, ge=6)
    max_display_messages: int = Field(default=20, ge=1, le=1000)
    refresh_per_second: int = Field(default=4, ge=1, le=60)
    message_overflow: Literal["crop", "ellipsis", "fold"] = "ellipsis"


class StdlibToLoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe()
        depth = 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_LOGGING_CONFIG = LoggingConfig()
_BASE_HANDLER_ID: int | None = None
_BRIDGE_INSTALLED = False
_ACTIVE_DISPLAY_SESSION: str | None = None
_STATE_LOCK = Lock()


def _is_in_jupyter() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        config = getattr(ip, "config", None) if ip is not None else None
        return ip is not None and config is not None and "IPKernelApp" in config
    except (ImportError, AttributeError):
        return False


def _normalize_level(level: Any) -> LogLevel:
    if isinstance(level, str) and level in VALID_LOG_LEVELS:
        return level  # type: ignore[return-value]
    return "INFO"


def _normalize_message(message: str) -> str:
    compact_message = " ".join(message.split())
    return compact_message or "-"


def _base_sink_filter(record: dict[str, Any]) -> bool:
    del record
    return _ACTIVE_DISPLAY_SESSION is None


def _display_sink_filter(session_id: str) -> Callable[[dict[str, Any]], bool]:
    def _filter(record: dict[str, Any]) -> bool:
        del record
        return _ACTIVE_DISPLAY_SESSION == session_id

    return _filter


def configure_logging(
    *,
    level: str | None = None,
    console_width: int | None = None,
    console_height: int | None = None,
    max_display_messages: int | None = None,
    refresh_per_second: int | None = None,
    message_overflow: Literal["crop", "ellipsis", "fold"] | None = None,
) -> LoggingConfig:
    updates: dict[str, Any] = {}
    if level is not None:
        updates["level"] = level
    if console_width is not None:
        updates["console_width"] = console_width
    if console_height is not None:
        updates["console_height"] = console_height
    if max_display_messages is not None:
        updates["max_display_messages"] = max_display_messages
    if refresh_per_second is not None:
        updates["refresh_per_second"] = refresh_per_second
    if message_overflow is not None:
        updates["message_overflow"] = message_overflow

    global _LOGGING_CONFIG
    _LOGGING_CONFIG = _LOGGING_CONFIG.model_copy(update=updates)

    if _BASE_HANDLER_ID is not None:
        ensure_logging(force=True)

    return _LOGGING_CONFIG.model_copy(deep=True)


def get_logging_config() -> LoggingConfig:
    return _LOGGING_CONFIG.model_copy(deep=True)


def create_console(*, width: int | None = None, height: int | None = None) -> Console:
    config = get_logging_config()
    return Console(
        width=config.console_width if width is None else width,
        height=config.console_height if height is None else height,
    )


def create_progress(
    *, console: Console | None = None, disable: bool = False
) -> Progress:
    progress_kwargs: dict[str, Any] = {"disable": disable}
    if console is not None:
        progress_kwargs["console"] = console

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        **progress_kwargs,
    )


def install_stdlib_bridge() -> None:
    global _BRIDGE_INSTALLED
    if _BRIDGE_INSTALLED:
        return

    scloop_logger = logging.getLogger("scloop")
    scloop_logger.handlers = [StdlibToLoguruHandler()]
    scloop_logger.setLevel(logging.NOTSET)
    scloop_logger.propagate = False

    _BRIDGE_INSTALLED = True


def ensure_logging(*, force: bool = False) -> None:
    global _BASE_HANDLER_ID

    with _STATE_LOCK:
        if _BASE_HANDLER_ID is not None and not force:
            install_stdlib_bridge()
            return

        if _BASE_HANDLER_ID is not None:
            logger.remove(_BASE_HANDLER_ID)
            _BASE_HANDLER_ID = None
        else:
            logger.remove()

        config = get_logging_config()
        _BASE_HANDLER_ID = logger.add(
            sys.stderr,
            level=config.level,
            format=DEFAULT_LOG_FORMAT,
            colorize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            filter=_base_sink_filter,
        )
        install_stdlib_bridge()


class LogEntry(BaseModel):
    timestamp: float = Field(ge=0)
    level: LogLevel
    message: str
    icon: str = ""


class LogCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    maxlen: int = Field(default=20, ge=1)
    messages: deque[LogEntry] = Field(default_factory=lambda: deque(maxlen=20))
    total_counts: dict[str, int] = Field(default_factory=dict)
    dropped_messages: int = 0
    _update_callback: Callable[[], None] | None = None

    def model_post_init(self, __context: Any) -> None:
        del __context
        if len(self.messages) == 0:
            self.messages = deque(maxlen=self.maxlen)
        elif self.messages.maxlen != self.maxlen:
            self.messages = deque(self.messages, maxlen=self.maxlen)

    def append(self, message: Any) -> None:
        record = getattr(message, "record", None)

        if record is not None:
            level = _normalize_level(record["level"].name)
            icon = getattr(record["level"], "icon", "")
            msg = record["message"]
            timestamp = float(record["time"].timestamp())
        else:
            level = "INFO"
            icon = ""
            msg = str(message).strip()
            timestamp = time.time()

        if len(self.messages) == self.maxlen:
            self.dropped_messages += 1

        entry = LogEntry(
            timestamp=timestamp,
            level=level,
            message=_normalize_message(str(msg)),
            icon=str(icon).strip(),
        )
        self.messages.append(entry)
        self.total_counts[level] = self.total_counts.get(level, 0) + 1

        if self._update_callback is not None:
            self._update_callback()

    def get_messages(self) -> list[LogEntry]:
        return list(self.messages)

    def get_total_counts(self) -> dict[str, int]:
        return self.total_counts.copy()

    def clear(self) -> None:
        self.messages.clear()
        self.total_counts.clear()
        self.dropped_messages = 0


class LogDisplay(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    maxlen: int | None = Field(default=None)
    refresh_per_second: int | None = Field(default=None)
    console_width: int | None = Field(default=None, ge=20)
    console_height: int | None = Field(default=None, ge=6)
    message_overflow: Literal["crop", "ellipsis", "fold"] | None = None
    transient: bool | None = None
    cache: LogCache = Field(default_factory=lambda: LogCache())
    console: Console | None = None
    progress: Progress | None = None
    _live: Live | None = None
    _handler_id: int | None = None
    _layout: Layout | None = None
    _session_id: str | None = None
    _last_refresh_at: float = 0.0
    _pending_refresh: bool = False
    _in_jupyter: bool = False

    def model_post_init(self, __context: Any) -> None:
        del __context
        config = get_logging_config()
        resolved_maxlen = self.maxlen or config.max_display_messages
        resolved_refresh = self.refresh_per_second or config.refresh_per_second

        self.maxlen = resolved_maxlen
        self.refresh_per_second = resolved_refresh
        if self.message_overflow is None:
            self.message_overflow = config.message_overflow
        if self.console is None:
            self.console = create_console(
                width=self.console_width,
                height=self.console_height,
            )

        self.console_width = self.console.width
        self.console_height = self.console.height
        self.cache = LogCache(maxlen=resolved_maxlen)

    def _summary_text(self) -> str | None:
        total_messages = sum(self.cache.total_counts.values())
        if total_messages == 0:
            return None

        summary = f"showing {len(self.cache.messages)}/{total_messages} messages"
        if self.cache.dropped_messages > 0:
            summary += f" | {self.cache.dropped_messages} older hidden"
        return summary

    def _create_table(self) -> Panel:
        table = Table(
            show_header=True, header_style="bold magenta", box=None, expand=True
        )
        table.add_column("Time", style="dim", width=8, no_wrap=True)
        table.add_column("Level", width=8, no_wrap=True)
        table.add_column(
            "Message",
            no_wrap=True,
            overflow=self.message_overflow,
        )

        for entry in self.cache.get_messages():
            dt = datetime.fromtimestamp(entry.timestamp)
            level_style = LEVEL_STYLES.get(entry.level, "white")
            table.add_row(
                dt.strftime("%H:%M:%S"),
                f"[{level_style}]{entry.level}[/{level_style}]",
                entry.message,
            )

        return Panel(
            table,
            title="[bold blue]Pipeline Logs",
            subtitle=self._summary_text(),
            border_style="blue",
        )

    def _create_layout(self) -> Layout:
        if self.progress is None:
            layout = Layout(name="logs")
            layout.update(self._create_table())
            return layout

        progress_height = 4
        log_height = self.maxlen + 4
        if self.console_height is not None:
            log_height = min(log_height, max(self.console_height - progress_height, 8))

        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=progress_height),
            Layout(name="logs", size=log_height),
        )
        layout["progress"].update(
            Panel(self.progress, title="[bold green]Progress", border_style="green")
        )
        layout["logs"].update(self._create_table())
        return layout

    def __enter__(self) -> LogCache:
        ensure_logging()

        self._in_jupyter = _is_in_jupyter()
        self._session_id = uuid.uuid4().hex
        with _STATE_LOCK:
            global _ACTIVE_DISPLAY_SESSION
            _ACTIVE_DISPLAY_SESSION = self._session_id

        self._handler_id = logger.add(
            self.cache.append,
            level="TRACE",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            filter=_display_sink_filter(self._session_id),
        )
        self.cache._update_callback = self.update
        self._layout = self._create_layout()
        self._live = Live(
            self._layout,
            console=self.console,
            auto_refresh=False,
            transient=False if self.transient is None else self.transient,
            vertical_overflow="crop",
        )
        self._live.__enter__()
        self.update(force=True)
        return self.cache

    def __exit__(self, *args: Any) -> None:
        self.cache._update_callback = None

        if self._handler_id is not None:
            logger.complete()
            self.update(force=True)
            logger.remove(self._handler_id)
            self._handler_id = None

        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None

        with _STATE_LOCK:
            global _ACTIVE_DISPLAY_SESSION
            if _ACTIVE_DISPLAY_SESSION == self._session_id:
                _ACTIVE_DISPLAY_SESSION = None
        self._session_id = None

    def update(self, *, force: bool = False) -> None:
        if self._live is None or self._layout is None:
            return

        refresh_interval = 1 / self.refresh_per_second
        now = time.monotonic()
        if not force and now - self._last_refresh_at < refresh_interval:
            self._pending_refresh = True
            return

        self._pending_refresh = False
        self._last_refresh_at = now

        if self.progress is not None:
            self._layout["logs"].update(self._create_table())
            self._live.refresh()
            return

        self._live.update(self._create_table(), refresh=True)
