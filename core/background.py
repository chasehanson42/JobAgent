"""Run long jobs (model download, resume reading, scoring, writing) off the page thread.

Streamlit re-runs the page script on every click and abandons whatever the
previous run was doing. Work started here lives in its own thread instead, so
it keeps going while the user clicks around; pages only read its progress.

One task runs at a time: they all share the single local AI model, so
running two would only make both slower.
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
    status: str = "running"            # running | done | error
    progress: float | None = None      # 0..1, None = unknown
    message: str = ""
    result: Any = None
    error: BaseException | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def running(self) -> bool:
        return self.status == "running"

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
                task.status = "done"
            except BaseException as exc:  # noqa: BLE001 - reported to the page
                task.error = exc
                task.status = "error"
            finally:
                task.finished = time.time()

        threading.Thread(target=run, name=f"task-{key}", daemon=True).start()
        return task

    def get(self, key: str) -> Task | None:
        return self._tasks.get(key)

    def running(self) -> Task | None:
        return next((t for t in self._tasks.values() if t.running), None)

    def latest(self) -> Task | None:
        return max(self._tasks.values(), key=lambda t: t.started, default=None)

    def clear(self, key: str) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task and not task.running:
                del self._tasks[key]

    def wait(self, timeout: float = 30) -> None:
        """Block until nothing is running (used by tests)."""
        deadline = time.time() + timeout
        while self.running() and time.time() < deadline:
            time.sleep(0.02)


# One manager per app process; Streamlit keeps imported modules across reruns.
MANAGER = TaskManager()
