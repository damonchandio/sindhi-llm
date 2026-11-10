@echo off
rem sindhi-llm v4: chat tuning again on CLEANED data (Wikipedia "References" endings cut, looping answers dropped),
rem starting from the Sindhi model, not from v3. Then a blind A/B: v3 vs v4, both with the new repeat blocker.
rem Double-click once; safe to close and double-click again (it resumes). About 5 hours. Logged to run.log.
cd /d "%~dp0"
echo ==== chat2 %date% %time% ==== >> run.log
call .venv\Scripts\activate
set "B=--base Qwen/Qwen3-4B --4bit"
if not exist results\chat-v3-4b.json ( python -m sindhi.chat_eval %B% --adapter out/lora-chat-4b --name chat-v3-4b >> run.log 2>&1 || goto :err )
echo Chat training on the cleaned data (about 4-5 hours). Safe to close; double-click again to continue.
if not exist out\lora-chat2-4b\adapter_model.safetensors ( python -m sindhi.chat_train --adapter out/lora-4b --out out/ckpt-chat2-4b >> run.log 2>&1 || goto :err )
if not exist out\lora-chat2-4b\adapter_model.safetensors ( python -m sindhi.chat_train --export --out out/ckpt-chat2-4b >> run.log 2>&1 || goto :err )
if not exist results\chat-v4-4b.json ( python -m sindhi.chat_eval %B% --adapter out/lora-chat2-4b --name chat-v4-4b >> run.log 2>&1 || goto :err )
if not exist results\belebele-mc-lora-chat2-4b.json ( python -m sindhi.belebele %B% --mc --adapter out/lora-chat2-4b --name lora-chat2-4b >> run.log 2>&1 || goto :err )
python -m sindhi.compare chat-v3-4b chat-v4-4b >> run.log 2>&1 || goto :err
start results\compare-chat-v4-4b.html
echo All done. On the A/B page "trained" means the new v4.
pause
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (tell Claude).
pause
exit /b 1
