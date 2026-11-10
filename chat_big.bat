@echo off
rem sindhi-llm v3: chat tuning. Double-click once; safe to close and double-click again (finished steps are
rem skipped, training resumes from its last save). Everything is logged to run.log.
rem   1. quick tests (~30 min): reading test with options A-D, for the original Qwen3-4B, last night's Sindhi model,
rem      and the original in English (the ceiling); plus the original's answers to 20 Sindhi requests
rem   2. chat training (~4-5 h)
rem   3. the same tests on the chat model, then the A/B page opens
rem   4. last: the reading test on the untrained Qwen3-8B (downloads ~16 GB) to see if a bigger model is worth it
cd /d "%~dp0"
echo ==== chat_big %date% %time% ==== >> run.log
call .venv\Scripts\activate
pip install -q -r requirements.txt
set "B=--base Qwen/Qwen3-4B --4bit"
echo Quick tests first (about 30 minutes)...
if not exist results\belebele-mc-base-4b.json ( python -m sindhi.belebele %B% --mc --name base-4b >> run.log 2>&1 || goto :err )
if not exist results\belebele-mc-lora-4b.json ( python -m sindhi.belebele %B% --mc --adapter out/lora-4b --name lora-4b >> run.log 2>&1 || goto :err )
if not exist results\belebele-mc-eng_Latn-base-4b.json ( python -m sindhi.belebele %B% --mc --lang eng_Latn --name base-4b >> run.log 2>&1 || goto :err )
if not exist results\chat-base-4b.json ( python -m sindhi.chat_eval %B% --name chat-base-4b >> run.log 2>&1 || goto :err )
echo Chat training (about 4-5 hours). Safe to close; double-click again to continue.
if not exist out\lora-chat-4b\adapter_model.safetensors ( python -m sindhi.chat_train >> run.log 2>&1 || goto :err )
if not exist out\lora-chat-4b\adapter_model.safetensors ( python -m sindhi.chat_train --export >> run.log 2>&1 || goto :err )
if not exist results\chat-lora-chat-4b.json ( python -m sindhi.chat_eval %B% --adapter out/lora-chat-4b --name chat-lora-chat-4b >> run.log 2>&1 || goto :err )
if not exist results\belebele-mc-lora-chat-4b.json ( python -m sindhi.belebele %B% --mc --adapter out/lora-chat-4b --name lora-chat-4b >> run.log 2>&1 || goto :err )
python -m sindhi.compare chat-base-4b chat-lora-chat-4b >> run.log 2>&1 || goto :err
start results\compare-chat-lora-chat-4b.html
echo Last: the bigger Qwen3-8B on the reading test (downloads about 16 GB)...
if not exist results\belebele-mc-base-8b.json ( python -m sindhi.belebele --base Qwen/Qwen3-8B --4bit --mc --name base-8b >> run.log 2>&1 || goto :err )
echo All done.
pause
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (tell Claude).
pause
exit /b 1
