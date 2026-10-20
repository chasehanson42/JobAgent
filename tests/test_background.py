import threading

import pytest

from core.background import TaskBusy, TaskManager


def test_task_runs_in_background_and_reports_progress():
    mgr = TaskManager()
    gate = threading.Event()

    def work(task):
        task.update(0.5, "halfway")
        gate.wait(5)
        return 42

    task = mgr.start("k", "Work", work)
    assert task.running and mgr.running() is task
    with pytest.raises(TaskBusy, match="Work"):
        mgr.start("other", "Other", lambda t: None)
    gate.set()
    mgr.wait()
    assert task.status == "done" and task.result == 42
    assert task.progress == 0.5 and task.message == "halfway" and task.finished


def test_task_errors_are_captured_not_raised():
    mgr = TaskManager()
    task = mgr.start("k", "Boom", lambda t: 1 / 0)
    mgr.wait()
    assert task.status == "error" and isinstance(task.error, ZeroDivisionError)
    # A finished task doesn't block the next one, and restarting a key replaces it.
    again = mgr.start("k", "Again", lambda t: "ok")
    mgr.wait()
    assert mgr.get("k") is again and again.result == "ok"
    assert mgr.latest() is again


def test_progress_is_clamped_and_clear_keeps_running_tasks():
    mgr = TaskManager()
    gate = threading.Event()
    task = mgr.start("k", "Work", lambda t: (t.update(7), gate.wait(5)))
    mgr.clear("k")
    assert mgr.get("k") is task          # running tasks can't be cleared
    gate.set()
    mgr.wait()
    assert task.progress == 1.0
    mgr.clear("k")
    assert mgr.get("k") is None
