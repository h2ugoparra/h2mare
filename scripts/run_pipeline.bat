@echo off
REM Run the h2mare pipeline from Windows Task Scheduler.
REM Location-independent: cd to the repo root (one level up from scripts\)
REM so config.yaml and .env are picked up from the working directory.

setlocal
cd /d "%~dp0.."

REM Ensure the log directory exists (it is otherwise git-ignored).
if not exist "logs" mkdir "logs"

REM Timestamped start marker, then run. Adjust the command/flags as needed.
echo ==== run started %DATE% %TIME% ==== >> "logs\pipeline.log"
uv run h2mare run >> "logs\pipeline.log" 2>&1
echo ==== run finished %DATE% %TIME% (exit %ERRORLEVEL%) ==== >> "logs\pipeline.log"

endlocal
