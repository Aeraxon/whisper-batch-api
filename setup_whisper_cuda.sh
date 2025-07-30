#!/bin/bash
# Whisper CUDA Setup Script
VENV_PATH="/home/user/whisper-batch-api/venv/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${VENV_PATH}/nvidia/cublas/lib:${VENV_PATH}/nvidia/cudnn/lib:${VENV_PATH}/ctranslate2.libs:/usr/local/cuda/lib64"
echo "✅ Whisper CUDA paths set"
