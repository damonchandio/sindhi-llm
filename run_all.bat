@echo off
rem sindhi-llm: one click on a Windows laptop with an NVIDIA GPU. Makes a venv, installs CUDA PyTorch,
rem downloads the data, scores the base model, trains the LoRA (about an hour on an 8 GB card), scores again,
rem and builds the A/B page. Safe to re-run: finished steps are skipped. Everything is logged to run.log.
cd /d "%~dp0"
echo ==== %date% %time% ==== >> run.log
if not exist .venv ( py -3.11 -m venv .venv || py -3 -m venv .venv )
call .venv\Scripts\activate
rem CUDA 12.8 build: the older cu124 wheels have no kernels for RTX 50-series (Blackwell) cards
python -c "import torch,sys; sys.exit(0 if torch.version.cuda and torch.version.cuda>='12.8' else 1)" 2>nul || pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE, CPU only')" >> run.log 2>&1
if not exist data\train.txt ( python -m sindhi.data 2>&1 >> run.log || goto :err )
if not exist results\base.json ( python -m sindhi.evaluate >> run.log 2>&1 || goto :err )
if not exist out\lora\adapter_model.safetensors ( python -m sindhi.train >> run.log 2>&1 || goto :err )
if not exist results\lora.json ( python -m sindhi.evaluate --adapter out/lora >> run.log 2>&1 || goto :err )
python -m sindhi.compare >> run.log 2>&1 || goto :err
echo All done. Opening the A/B page.
start results\compare.html
pause
exit /b 0
:err
echo.
echo A step failed. The details are in run.log (send it to Claude).
pause
exit /b 1
