@echo off
REM Windows setup script for the model benchmarking suite

echo Setting up Hugging Face Model Benchmark Suite...

REM Create directories
if not exist "models" mkdir models
if not exist "results" mkdir results
if not exist "logs" mkdir logs

REM Install requirements
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Set environment variables for better performance
set TOKENIZERS_PARALLELISM=false
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TRANSFORMERS_OFFLINE=0

REM Check system capabilities
echo Checking system capabilities...
python -c "import torch; import sys; print(f'Python version: {sys.version}'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo Setup complete! You can now run:
echo   python main_orchestrator.py

REM Optional: Run a quick system check
echo Running quick system resource check...
python -c "from resource_monitor import quick_resource_check; import json; print(json.dumps(quick_resource_check(), indent=2))"

echo Setup finished successfully!
pause