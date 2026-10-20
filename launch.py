"""Start the JobAgent web app and open it in the browser. Used by the Start launchers."""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://localhost:8501"


def is_up() -> bool:
    try:
        with urllib.request.urlopen(URL + "/_stcore/health", timeout=1):
            return True
    except OSError:
        return False


def main() -> int:
    if is_up():
        print("JobAgent is already running. Opening it in your browser.")
        webbrowser.open(URL)
        return 0

    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=ROOT)
    for _ in range(90):
        if proc.poll() is not None:
            print("\nJobAgent stopped unexpectedly. See the messages above.")
            return proc.returncode or 1
        if is_up():
            webbrowser.open(URL)
            break
        time.sleep(1)

    print("\n  JobAgent is open in your web browser at", URL)
    print("  Keep this window open while you use it. Close the window to stop JobAgent.\n")
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
