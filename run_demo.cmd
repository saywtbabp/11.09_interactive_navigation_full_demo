@echo off
setlocal
cd /d "%~dp0"
set "DEMO_PYTHON=G:\anaconda3\envs\mujoco\python.exe"
if defined DRONE_PYTHON set "DEMO_PYTHON=%DRONE_PYTHON%"
if not exist "%DEMO_PYTHON%" (
  echo Please select the existing Conda interpreter in PyCharm and run main.py.
  pause
  exit /b 1
)
"%DEMO_PYTHON%" main.py %*
if errorlevel 1 pause
endlocal
