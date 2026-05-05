from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import FrameType
from typing import Any, NoReturn
import os
import signal
import threading

import jax
from jax.errors import JaxRuntimeError


def default_runtime_interrupt_signals() -> tuple[signal.Signals, ...]:
    """Return terminal signals that request graceful runtime cancellation."""

    names = ("SIGINT", "SIGTERM", "SIGTSTP")
    return tuple(
        signal.Signals(getattr(signal, name)) for name in names if hasattr(signal, name)
    )


class RuntimeInterrupted(KeyboardInterrupt):
    """Raised when a terminal signal requests VerCOR runtime cancellation."""

    def __init__(self, signum: int | None, label: str = "runtime") -> None:
        self.signum = signum
        self.signal_name = _signal_name(signum)
        self.label = label
        super().__init__(f"VerCOR runtime interrupted by {self.signal_name} at {label}")


class RuntimeInterruptController:
    """Coordinate terminal-signal interruption across host and scanned runtimes."""

    def __init__(
        self,
        signals: Sequence[int | signal.Signals] | None = None,
    ) -> None:
        selected_signals = (
            default_runtime_interrupt_signals() if signals is None else signals
        )
        self._signals = tuple(int(signum) for signum in selected_signals)
        self._requested_signal: int | None = None
        self._previous_handlers: dict[int, Any] = {}
        self._wakeup = _SignalWakeupPipe(self._signals)
        self._scope_depth = 0

    @property
    def requested_signal(self) -> int | None:
        """Return the pending signal number, if cancellation was requested."""

        return self._requested_signal

    @contextmanager
    def signal_scope(self) -> Iterator[None]:
        """Install terminal-signal handlers for one outer runtime call."""

        if self._scope_depth > 0:
            self._scope_depth += 1
            try:
                yield
            finally:
                self._scope_depth -= 1
            return

        self.clear()
        if not _can_install_signal_handlers():
            try:
                yield
            finally:
                self.clear()
            return

        self._previous_handlers = {}
        try:
            for signum in self._signals:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self.request_from_signal)
            self._wakeup.install()
            self._scope_depth = 1
            yield
        finally:
            self._scope_depth = 0
            self._wakeup.restore()
            for signum, handler in self._previous_handlers.items():
                signal.signal(signum, handler)
            self._previous_handlers = {}
            self.clear()

    def request(self, signum: int | signal.Signals) -> None:
        """Record that ``signum`` requested runtime cancellation."""

        if self._requested_signal is None:
            self._requested_signal = int(signum)

    def request_from_signal(self, signum: int, frame: FrameType | None) -> None:
        """Signal-handler entrypoint that records a cancellation request."""

        _ = frame
        self.request(signum)

    def clear(self) -> None:
        """Clear any pending cancellation request."""

        self._requested_signal = None
        self._wakeup.drain()

    def checkpoint(self, label: str = "runtime") -> None:
        """Raise ``RuntimeInterrupted`` when a terminal signal is pending."""

        self._request_from_wakeup_fd()
        if self._requested_signal is not None:
            raise RuntimeInterrupted(self._requested_signal, label)

    def _request_from_wakeup_fd(self) -> None:
        """Promote pending wakeup-fd bytes into a runtime interruption request."""

        signum = self._wakeup.drain()
        if signum is not None:
            self.request(signum)

    def scanned_checkpoint(
        self,
        label: str = "scanned runtime",
        token: Any | None = None,
    ) -> None:
        """Insert a callback checkpoint into a traced scanned runtime."""

        def emit(*_args: Any) -> None:
            self.checkpoint(label)

        if token is None:
            jax.debug.callback(emit, ordered=True)
            return
        jax.debug.callback(emit, token, ordered=True)

    def raise_if_jax_callback_interrupted(
        self,
        error: JaxRuntimeError,
        label: str = "compiled scanned runtime",
    ) -> NoReturn:
        """Translate interrupt callback failures back to ``KeyboardInterrupt``."""

        message = str(error)
        callback_failed = "callback" in message.lower()
        controller_interrupt = "RuntimeInterrupted" in message or (
            self._requested_signal is not None and "KeyboardInterrupt" in message
        )
        if callback_failed and controller_interrupt:
            signal_number = self._requested_signal
            raise RuntimeInterrupted(signal_number, label) from error
        raise error


def _can_install_signal_handlers() -> bool:
    return threading.current_thread() is threading.main_thread()


class _SignalWakeupPipe:
    """Small nonblocking pipe used by ``signal.set_wakeup_fd``."""

    def __init__(self, signals: Sequence[int]) -> None:
        self._signals = frozenset(signals)
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._previous_fd: int | None = None

    def install(self) -> None:
        """Install this pipe as the process wakeup fd."""

        if self._read_fd is not None:
            return

        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(read_fd, False)
            os.set_blocking(write_fd, False)
            self._previous_fd = signal.set_wakeup_fd(
                write_fd,
                warn_on_full_buffer=False,
            )
        except BaseException:
            _close_fd(read_fd)
            _close_fd(write_fd)
            self._previous_fd = None
            raise

        self._read_fd = read_fd
        self._write_fd = write_fd

    def restore(self) -> None:
        """Restore the previous process wakeup fd and close this pipe."""

        read_fd = self._read_fd
        write_fd = self._write_fd
        previous_fd = self._previous_fd
        self._read_fd = None
        self._write_fd = None
        self._previous_fd = None

        try:
            if previous_fd is not None:
                signal.set_wakeup_fd(previous_fd, warn_on_full_buffer=False)
        finally:
            _close_fd(read_fd)
            _close_fd(write_fd)

    def drain(self) -> int | None:
        """Return the first configured signal number drained from the pipe."""

        if self._read_fd is None:
            return None

        requested_signal: int | None = None
        while True:
            try:
                data = os.read(self._read_fd, 4096)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            except OSError:
                break
            if not data:
                break

            for signum in data:
                if requested_signal is None and signum in self._signals:
                    requested_signal = signum

        return requested_signal


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        return


def _signal_name(signum: int | None) -> str:
    if signum is None:
        return "an unknown signal"
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"
