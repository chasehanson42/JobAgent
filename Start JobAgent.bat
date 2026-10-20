@echo off
rem Double-click this file to start JobAgent. First run sets everything up.
setlocal EnableExtensions
title JobAgent
cd /d "%~dp0"
echo.
echo   ==========================================
echo     JobAgent is starting...
echo   ==========================================
echo.

rem ---- 1. Find Python 3.11 or newer --------------------------------------
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY goto :no_python

rem ---- 2. One-time setup: private Python environment for JobAgent --------
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo   First-time setup. This takes a few minutes and only happens once.
  echo.
  %PY% -m venv .venv
  if errorlevel 1 goto :setup_failed
)

fc /b requirements.txt .venv\installed-requirements.txt >nul 2>nul
if errorlevel 1 (
  echo   Installing JobAgent's components...
  "%VPY%" -m pip install --disable-pip-version-check -q --upgrade pip
  "%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 goto :setup_failed
  copy /y requirements.txt .venv\installed-requirements.txt >nul
)

rem Job-site search add-on: optional, tried once. Pasting jobs works without it.
if not exist ".venv\scraping-attempted.txt" (
  echo   Installing the job-site search add-on...
  "%VPY%" -m pip install --disable-pip-version-check -q -r requirements-scraping.txt
  if errorlevel 1 (
    echo   [!] The job-site search add-on could not be installed.
    echo       You can still paste jobs into JobAgent by hand.
  )
  echo attempted> ".venv\scraping-attempted.txt"
)

rem ---- 3. Start Ollama if it is installed but not running ----------------
curl.exe -s -m 2 http://localhost:11434/api/version >nul 2>nul
if errorlevel 1 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
  echo   Starting Ollama...
  start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
)

rem ---- 4. Run the app ----------------------------------------------------
"%VPY%" launch.py
if errorlevel 1 pause
exit /b

:no_python
echo   JobAgent needs Python 3.11 or newer, which isn't installed yet.
echo.
echo   1. Your browser will open the Python download page.
echo   2. Download and run the installer.
echo      IMPORTANT: tick "Add python.exe to PATH" on the first screen.
echo   3. When it finishes, double-click "Start JobAgent" again.
echo.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:setup_failed
echo.
echo   [!] Setup didn't finish. Check your internet connection and try again.
echo       If it keeps failing, send a screenshot of this window to whoever
echo       shared JobAgent with you.
echo.
pause
exit /b 1
