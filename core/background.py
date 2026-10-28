"""Run long jobs (model download, resume reading, scoring, writing) off the page thread.

Streamlit re-runs the page script on every click and abandons whatever the
previous run was doing. Work started here lives in its own thread instead, so
it keeps going while the user clicks around; pages only read its progress.

One task runs at a time: they all share the single local AI model, so
running two would only make both slower.

Any task can be stopped. Stopping sets a flag the task checks between steps
and runs the task's stop hooks (e.g. cutting off an AI call in progress). A
stopping task no longer blocks new ones, and keeps whatever it already saved.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class TaskBusy(Exception):
    def __init__(self, running: "Task"):
        super().__init__(f"Please wait until “{running.label}” finishes.")
        self.running = running


@dataclass
class Task:
    key: str
    label: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"            # running | done | error | stopped
    progress: float | None = None      # 0..1, None = unknown
    message: str = ""
    result: Any = None
    error: BaseException | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _hooks: list[Callable[[], Any]] = field(default_factory=list, repr=False)

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def on_stop(self, hook: Callable[[], Any]) -> None:
        """Run `hook` when the user presses Stop (immediately, if already stopped)."""
        self._hooks.append(hook)
        if self.stop_requested:
            hook()

    def request_stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self.message = "Stopping..."
        for hook in list(self._hooks):
            try:
                hook()
            except Exception:  # noqa: BLE001 - stopping must never raise
                pass

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}

    def start(self, key: str, label: str, fn: Callable[[Task], Any]) -> Task:
        """Run fn(task) in a thread. Raises TaskBusy if another task is running."""
        with self._lock:
            busy = self.running()
            if busy:
                raise TaskBusy(busy)
            task = Task(key, label)
            self._tasks[key] = task

        def run() -> None:
            try:
                task.result = fn(task)
                task.status = "stopped" if task.stop_requested else "done"
            except BaseException as exc:  # noqa: BLE001 - reported to the page
                if task.stop_requested:
                    task.status = "stopped"     # the error is just the Stop cutting a call off
                else:
                    task.error = exc
                    task.status = "error"
            finally:
                task.finished = time.time()

        threading.Thread(target=run, name=f"task-{key}", daemon=True).start()
        return task

    def get(self, key: str) -> Task | None:
        return self._tasks.get(key)

    def running(self) -> Task | None:
        """The task that blocks starting another one (a task being stopped doesn't)."""
        return next((t for t in self._tasks.values() if t.running and not t.stop_requested), None)

    def stop(self, key: str) -> None:
        task = self._tasks.get(key)
        if task is not None and task.running:
            task.request_stop()

    def latest(self) -> Task | None:
        return max(self._tasks.values(), key=lambda t: t.started, default=None)

    def clear(self, key: str) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task and not task.running:
                del self._tasks[key]

    def wait(self, timeout: float = 30) -> None:
        """Block until no task thread is still working (used by tests)."""
        deadline = time.time() + timeout
        while any(t.running for t in self._tasks.values()) and time.time() < deadline:
            time.sleep(0.02)


# One manager per app process; Streamlit keeps imported modules across reruns.
MANAGER = TaskManager()
