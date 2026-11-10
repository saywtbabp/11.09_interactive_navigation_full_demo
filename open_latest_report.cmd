@echo off
setlocal
set "REPORT_PATH=%~dp0report.html"
if not exist "%REPORT_PATH%" (
  echo Run main.py to generate a new report, or see README.md.
  pause
  exit /b 1
)
start "" "%REPORT_PATH%"
endlocal
