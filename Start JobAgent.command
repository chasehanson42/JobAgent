#!/bin/bash
# Double-click this file to start JobAgent on a Mac. First run sets everything up.
cd "$(dirname "$0")" || exit 1
echo
echo "  JobAgent is starting..."
echo

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "  JobAgent needs Python 3.11 or newer, which isn't installed yet."
  echo "  1. Your browser will open the Python download page. Install the macOS version."
  echo "  2. When it finishes, double-click \"Start JobAgent\" again."
  open "https://www.python.org/downloads/macos/"
  read -r -p "  Press Enter to close..."
  exit 1
fi

fail() {
  echo
  echo "  [!] Setup didn't finish. Check your internet connection and try again."
  echo "      If it keeps failing, send a screenshot of this window to whoever shared JobAgent with you."
  read -r -p "  Press Enter to close..."
  exit 1
}

VPY=".venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "  First-time setup. This takes a few minutes and only happens once."
  "$PY" -m venv .venv || fail
fi
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
  echo "  Installing JobAgent's components..."
  "$VPY" -m pip install --disable-pip-version-check -q --upgrade pip
  "$VPY" -m pip install --disable-pip-version-check -q -r requirements.txt || fail
  cp requirements.txt .venv/installed-requirements.txt
fi
if [ ! -f .venv/scraping-attempted.txt ]; then
  echo "  Installing the job-site search add-on..."
  "$VPY" -m pip install --disable-pip-version-check -q -r requirements-scraping.txt \
    || echo "  [!] The job-site search add-on could not be installed. You can still paste jobs by hand."
  echo attempted > .venv/scraping-attempted.txt
fi

if ! curl -s -m 2 http://localhost:11434/api/version >/dev/null 2>&1 && [ -d /Applications/Ollama.app ]; then
  echo "  Starting Ollama..."
  open -a Ollama
fi

"$VPY" launch.py || read -r -p "  Press Enter to close..."
