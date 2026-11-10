@echo off
rem sindhi-llm v5 on the 4B: chat tuning from the Sindhi model on train_v6 (MURI-IT cut to 2,000 + ~3,000 Sindhi
rem multiple-choice reading questions), learning rate decaying to zero. Then Belebele and a blind A/B: v4 vs v5.
rem Double-click once; safe to close and double-click again (it resumes). About 5 hours. Logged to run.log.
rem Let it finish: with the decaying learning rate, a run stopped halfway is weaker than v4's.
cd /d "%~dp0"
echo ==== chat3 %date% %time% ==== >> run.log
call .venv\Scripts\activate
set "B=--base Qwen/Qwen3-4B --4bit"
rem Groq key from Connor's settings (never printed). The question writer resumes where it stopped and ends at 3,000.
for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b "GROQ_API_KEY=" "..\..\.env"`) do set "GROQ_API_KEY=%%b"
if not exist data\chat\train_v6.jsonl (
  echo Writing Sindhi multiple-choice questions with Groq's free models ^(several hours; Groq's daily free limit may stop it, so double-click again tomorrow and it carries on^).
  python -m sindhi.mc_data --n 3000 >> run.log 2>&1 || goto :err
  echo Check data\chat\mc_sample.txt ^(50 rows^). Close this window if they look wrong; otherwise
  pause
  python -m sindhi.mc_data --merge >> run.log 2>&1 || goto :err
)
echo Chat training on v6 (about 4-5 hours). Safe to close; double-click again to continue.
if not exist out\lora-chat3-4b\adapter_model.safetensors ( python -m sindhi.chat_train --adapter out/lora-4b --data train_v6.jsonl --sched cosine --out out/ckpt-chat3-4b >> run.log 2>&1 || goto :err )
if not exist out\lora-chat3-4b\adapter_model.safetensors ( python -m sindhi.chat_train --export --out out/ckpt-chat3-4b >> run.log 2>&1 || goto :err )
if not exist results\belebele-mc-lora-chat3-4b.json ( python -m sindhi.belebele %B% --mc --adapter out/lora-chat3-4b --name lora-chat3-4b >> run.log 2>&1 || goto :err )
if not exist results\chat-v5-4b.json ( python -m sindhi.chat_eval %B% --adapter out/lora-chat3-4b --name chat-v5-4b >> run.log 2>&1 || goto :err )
python -m sindhi.compare chat-v4-4b chat-v5-4b >> run.log 2>&1 || goto :err
type results\belebele-mc-lora-chat3-4b.json
start results\compare-chat-v5-4b.html
echo All done. Belebele score above. On the A/B page "trained" means the new v5.
pause
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (tell Claude).
pause
exit /b 1
