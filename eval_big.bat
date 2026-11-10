@echo off
rem Scores the newest save of the big model and builds a blind A/B page against the untrained one.
rem Stop train_big.bat first (close its window): the GPU can't train and score at once.
cd /d "%~dp0"
echo ==== eval_big %date% %time% ==== >> run.log
call .venv\Scripts\activate
set "BASE=Qwen/Qwen3-4B"
set "NAME=4b"
set "Q=--4bit"
if exist out\ckpt-1.7b ( set "BASE=Qwen/Qwen3-1.7B" & set "NAME=1.7b" & set "Q=" )
python -m sindhi.train_big --export --out out/ckpt-%NAME% >> run.log 2>&1 || goto :err
python -m sindhi.evaluate --base %BASE% %Q% --adapter out/lora-%NAME% --name lora-%NAME% >> run.log 2>&1 || goto :err
python -m sindhi.belebele --base %BASE% %Q% --adapter out/lora-%NAME% --name lora-%NAME% >> run.log 2>&1 || goto :err
python -m sindhi.compare base-%NAME% lora-%NAME% >> run.log 2>&1 || goto :err
echo Done. Opening the A/B page.
start results\compare-lora-%NAME%.html
pause
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (tell Claude).
pause
exit /b 1
