@echo off
REM Start hanuman.ai with the voice pipeline actually working.
REM
REM Piper and the Nepali Whisper model live in WSL (.venv-wsl), not on Windows.
REM Starting the server from the Windows venv gives one whose /portal/voice-status
REM reports "piper is not installed on this host", so the portal hides the mic
REM (see checkVoice() in app/static/portal.html) and the agent never speaks —
REM a working-looking text-only call. Serving the same app from WSL on the same
REM port keeps one URL that always has voice: http://127.0.0.1:8000/portal
REM
REM First spoken turn is slow: Whisper and OpenVoice load on demand.

for /f "delims=" %%p in ('wsl wslpath -a "%~dp0."') do set "PROJ=%%p"
if "%PROJ%"=="" (
  echo Could not map this folder into WSL — is WSL installed?
  exit /b 1
)
echo Serving from WSL at %PROJ%
echo Portal: http://127.0.0.1:8000/portal    ^(Ctrl-C to stop^)
wsl -e bash -lc "cd '%PROJ%' && PATH=\"$PWD/.venv-wsl/bin:$PATH\" exec ./.venv-wsl/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
