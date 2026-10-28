"""Start the JobAgent web app and open it in the browser. Used by the Start launchers.

If a JobAgent is already running, it is reused only when it runs this exact
version. An older copy left running in the background (after an update, a
crash, or a window that didn't close cleanly) is stopped and replaced;
otherwise every start would reconnect to the old code.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8501
URL = f"http://localhost:{PORT}"
SERVER_INFO = ROOT / "data" / "server.json"


def is_up() -> bool:
    try:
        with urllib.request.urlopen(URL + "/_stcore/health", timeout=1):
            return True
    except OSError:
        return False


def current_version() -> str:
    text = (ROOT / "core" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else ""


def running_info() -> dict:
    try:
        return json.loads(SERVER_INFO.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_windows_netstat(output: str, port: int) -> int | None:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(f":{port}") \
                and parts[3].upper() == "LISTENING" and parts[4].isdigit():
            return int(parts[4])
    return None


def pid_on_port(port: int = PORT) -> int | None:
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True,
                                 timeout=10).stdout
            return parse_windows_netstat(out, port)
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                             text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def is_python(pid: int) -> bool:
    """Only ever stop a Python process (that's what JobAgent runs as)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout.lower()
        else:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True,
                                 timeout=10).stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False
    return "python" in out or "streamlit" in out


def stop(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=20)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(40):
        if not is_up():
            return
        time.sleep(0.25)


def replace_stale_server() -> bool:
    """Return True if a running JobAgent of this version can be reused."""
    info = running_info()
    version = current_version()
    if info.get("version") == version:
        return True
    pid = pid_on_port()
    if pid is None or not is_python(pid):
        print(f"\n  Another program is using port {PORT}, so JobAgent can't start.")
        print("  Restart your computer and try again.\n")
        raise SystemExit(1)
    print(f"  An older JobAgent ({info.get('version') or 'unknown version'}) is still running. "
          f"Replacing it with version {version}...")
    stop(pid)
    return False


def main() -> int:
    if is_up() and replace_stale_server():
        print("JobAgent is already running. Opening it in your browser.")
        webbrowser.open(URL)
        return 0

    SERVER_INFO.unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=ROOT)
    for _ in range(90):
        if proc.poll() is not None:
            print("\nJobAgent stopped unexpectedly. See the messages above.")
            return proc.returncode or 1
        if is_up():
            webbrowser.open(URL)
            break
        time.sleep(1)

    print(f"\n  JobAgent {current_version()} is open in your web browser at", URL)
    print("  Keep this window open while you use it. Close the window to stop JobAgent.\n")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
