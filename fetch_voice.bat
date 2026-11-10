@echo off
rem Downloads your phone recordings into data\voice (wavs + metadata.csv). Safe to run while training.
cd /d "%~dp0"
call .venv\Scripts\activate
python -m sindhi.voice
pause
