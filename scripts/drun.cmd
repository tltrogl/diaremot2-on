@echo off
setlocal
set "REPO=%~dp0.."
set "PYTHON=%REPO%\.balls\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Virtualenv not found at .\.balls\Scripts\python.exe
  exit /b 1
)
set "PYTHONPATH=%REPO%\src"
"%PYTHON%" -m diaremot.cli %*
exit /b %ERRORLEVEL%

