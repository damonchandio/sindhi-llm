@echo off
rem sindhi-llm v2: Qwen3-4B, 4-bit, ~50M tokens of Sindhi (1500 steps, about 1.5 days). Built to be stopped:
rem close this window any time, double-click again, and it
rem carries on from the last save (every 25 steps). Run eval_big.bat whenever you want a score.
rem Keep it plugged in. Windows setting: "When I close the lid" = Do nothing (plugged in).
cd /d "%~dp0"
echo ==== train_big %date% %time% ==== >> run.log
if not exist .venv ( py -3.11 -m venv .venv || py -3 -m venv .venv )
call .venv\Scripts\activate
python -c "import torch,sys; sys.exit(0 if torch.version.cuda and torch.version.cuda>='12.8' else 1)" 2>nul || pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
set "BASE=Qwen/Qwen3-4B"
set "NAME=4b"
set "Q=--4bit"
set "T="
rem fallback if 4-bit doesn't run on this GPU: a smaller model at full precision (still fits 8 GB)
python -c "import torch,bitsandbytes as b; l=b.nn.Linear4bit(64,64,compute_dtype=torch.bfloat16).cuda(); l(torch.randn(2,64,device='cuda',dtype=torch.bfloat16)); print('4-bit OK')" >> run.log 2>&1 || ( set "BASE=Qwen/Qwen3-1.7B" & set "NAME=1.7b" & set "Q=" & set "T=--no-4bit" & echo 4-bit failed, falling back to Qwen3-1.7B >> run.log )
echo Model: %BASE%   (progress is in run.log)
if not exist data\train.txt ( python -m sindhi.data >> run.log 2>&1 || goto :err )
if not exist data\corpus\stats.json ( echo Downloading and cleaning the Sindhi corpus, 20-40 min... & python -m sindhi.corpus --max-chars 2e8 >> run.log 2>&1 || goto :err )
if not exist data\tokens\meta.json ( echo Tokenising... & python -m sindhi.tokenize_corpus --base %BASE% >> run.log 2>&1 || goto :err )
if not exist results\base-%NAME%.json ( echo Scoring the untrained model... & python -m sindhi.evaluate --base %BASE% %Q% --name base-%NAME% >> run.log 2>&1 || goto :err )
if not exist results\belebele-base-%NAME%.json ( python -m sindhi.belebele --base %BASE% %Q% --name base-%NAME% >> run.log 2>&1 || goto :err )
echo Training. Safe to close this window any time; double-click again to continue.
python -m sindhi.train_big --base %BASE% %T% --out out/ckpt-%NAME% --batch 1 --accum 32 --max-steps 1500 >> run.log 2>&1 || goto :err
call eval_big.bat
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (tell Claude).
pause
exit /b 1
