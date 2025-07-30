# Whisper Batch Transcription API

A high-performance batch transcription service using OpenAI's Whisper models with flexible configuration, optimized for GPU acceleration and designed for processing thousands of audio files efficiently.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Aeraxon/whisper-batch-api.git
cd whisper-batch-api
python3.10 -m venv venv && source venv/bin/activate

# 2. Install (choose your GPU)
# Standard GPUs (RTX 40XX and older):
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# Blackwell GPUs (RTX 50XX):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies and start
pip install -r requirements.txt

# Apply container restart fix (important for LXC/Docker)
echo 'VENV_SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"' >> venv/bin/activate
echo 'export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/cublas/lib:${VENV_SITE_PACKAGES}/nvidia/cudnn/lib:${VENV_SITE_PACKAGES}/ctranslate2.libs:/usr/local/cuda/lib64"' >> venv/bin/activate

python whisper_api.py

# 4. Test transcription
curl -X POST "http://localhost:8000/transcribe/batch" -F "files=@your_audio.wav"
```

## Features

- 🚀 **High Performance**: Support for all Whisper models with automatic GPU optimization
- 🌍 **Smart Language Detection**: Automatic recognition with manual override option
- 📦 **Batch Processing**: Process up to 100 files per batch
- 📊 **Performance Metrics**: Automatic tracking with CSV export
- 🔧 **Flexible Configuration**: Central YAML configuration for all settings
- 🔄 **Lazy Loading**: Models loaded on-demand, auto-unloaded after inactivity
- 📁 **Automatic Archiving**: All transcripts saved with metadata
- 🖥️ **GPU Monitoring**: VRAM tracking and utilization metrics

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX A2000 12GB, RTX 5080)
- **RAM**: Minimum 16GB system RAM
- **Storage**: SSD recommended
- **CUDA**: Compatible GPU with CUDA Compute Capability 6.0+

## Performance & Metrics

The system automatically tracks comprehensive performance metrics for every batch and saves them to `./output/metrics.csv`. View real-time metrics:

```bash
# Real-time metrics via API
curl http://localhost:8000/metrics

# Check saved metrics
cat output/metrics.csv
```

**⚡ Blackwell GPU Performance:** RTX 50XX users get automatic performance optimization with mixed precision (int8_float16) for optimal speed.

## Supported Models

### **Large Models (Recommended):**
- `large-v3` - Latest and most accurate
- `large-v3-turbo` - Faster variant of v3  
- `large-v3-german` - **Optimized for German language**
- `large-v2` - Proven and stable

### **Smaller Models:**
- `medium`, `small`, `base`, `tiny` - Various speed/quality trade-offs

## Installation

### 1. System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install development tools
sudo apt install -y build-essential cmake git wget curl

# Install Python 3.10 and venv
sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Install audio system dependencies
sudo apt install -y ffmpeg
```

### 2. Verify NVIDIA Setup

Ensure your NVIDIA driver and CUDA toolkit are properly installed:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit  
nvcc --version
```

Both commands should work without errors before proceeding.

### 3A. Python Environment Setup (Standard GPUs)

```bash
# Clone the repository
git clone https://github.com/Aeraxon/whisper-batch-api.git
cd whisper-batch-api

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support FIRST
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 3B. Python Environment Setup (Blackwell GPUs - RTX 50XX) 🆕

**For RTX 5090, 5080, 5070 and other Blackwell architecture GPUs:**

```bash
# Clone the repository
git clone https://github.com/Aeraxon/whisper-batch-api.git
cd whisper-batch-api

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (native for Blackwell GPUs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 4A. CUDA Library Path Configuration (Standard)

**This step is essential for the system to work properly:**

```bash
# Add CUDA libraries to LD_LIBRARY_PATH permanently
echo '' >> ~/.bashrc
echo '# Whisper CUDA Libraries' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia/cublas/lib:$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia/cudnn/lib:$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/ctranslate2.libs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Reload bash configuration
source ~/.bashrc
```

### 4B. CUDA Library Path Configuration (Blackwell GPUs) 🆕

**For CUDA 12.8+ with Blackwell GPUs:**

```bash
# Add CUDA 12.8+ libraries to LD_LIBRARY_PATH
echo '' >> ~/.bashrc
echo '# Whisper CUDA Libraries for Blackwell GPUs' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia/cublas/lib:$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/nvidia/cudnn/lib:$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)/ctranslate2.libs:/usr/local/cuda-12.8/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Reload bash configuration
source ~/.bashrc
```

### 5. Container Restart Fix (LXC/Docker)

**Important for LXC/Docker users:** After container restarts, you may encounter `libcudnn_ops.so` errors. Here's the permanent fix:

```bash
# Add CUDA paths directly to venv activation (recommended)
echo '' >> venv/bin/activate
echo '# Whisper CUDA Libraries (Container Restart Fix)' >> venv/bin/activate
echo 'VENV_SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"' >> venv/bin/activate
echo 'export LD_LIBRARY_PATH="${VENV_SITE_PACKAGES}/nvidia/cublas/lib:${VENV_SITE_PACKAGES}/nvidia/cudnn/lib:${VENV_SITE_PACKAGES}/ctranslate2.libs:/usr/local/cuda/lib64"' >> venv/bin/activate
```

**Test the fix:**
```bash
deactivate
source venv/bin/activate
python -c "from faster_whisper import WhisperModel; print('✅ CUDA Fix Working!')"
```

**Create convenient start script:**
```bash
# Create start script in your home directory
cat > ~/start_whisper.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Whisper Batch API..."
cd /path/to/your/whisper-batch-api
source venv/bin/activate
echo "✅ CUDA paths set automatically"
python whisper_api.py
EOF

chmod +x ~/start_whisper.sh
```

> **Note:** Replace `/path/to/your/whisper-batch-api` with your actual project path.

### 6. Verify Installation

```bash
# Test CUDA availability
python -c "
import torch
from faster_whisper import WhisperModel
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU Compute capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')
print('Faster-Whisper import successful!')
"
```

**For Blackwell GPUs, you should see:**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5080  # or your specific Blackwell GPU
CUDA version: 12.8  # Native PyTorch CUDA 12.8 support
GPU Compute capability: (9, 0)  # Blackwell architecture
Faster-Whisper import successful!
```

### 7. Create Output Directory

```bash
# Create directory for transcripts and metrics
mkdir -p output
```

## Configuration

The system uses a central YAML configuration file (`whisper_config.yaml`) for all settings. Key options:

- **Model**: `default_model: "large-v2"` (switch between models)
- **Language**: `auto_detect: true` (automatic language detection)
- **Output**: `directory: "./output"` (where transcripts are saved)
- **Batch Size**: `batch_size: 4` (adjust for your GPU)

See `whisper_config.yaml` for all configuration options.

## Dependencies (requirements.txt)

```
fastapi==0.115.5
uvicorn==0.32.1
python-multipart==0.0.12
aiofiles==24.1.0
faster-whisper==1.1.0
pynvml==11.5.3
pyyaml==6.0.2
soundfile==0.12.1
```

## Usage

### Start the API Server

**Quick Start (with start script):**
```bash
# If you created the start script during installation:
~/start_whisper.sh
```

**Manual Start:**
```bash
# Always activate venv first (sets CUDA paths automatically)
source venv/bin/activate
python whisper_api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Simple Batch Transcription
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@audio1.wav" \
     -F "files=@audio2.mp3"
```

#### With Language Override
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@english_audio.wav" \
     -F "language=en"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

#### View Performance Metrics
```bash
curl http://localhost:8000/metrics
```

### Supported Audio Formats

WAV, MP3, M4A, OGG, FLAC, AAC

## Output Files

### Transcripts
Automatically saved to `./output/transcript_XXXXXX.txt` with metadata:

```
# Whisper Transcription #000001
# Original file: audio.wav
# Audio length: 6.25 minutes
# Processing time: 54.30s
# Real-time factor: 6.9x
# Model: large-v2
# Language mode: auto-detect
# Detected language: de (confidence: 0.98)
# =====================================

[Transcript content here]
```

### Performance Metrics
Saved to `./output/metrics.csv` with processing times, throughput, GPU utilization, and capacity planning data.

## Troubleshooting

### Container Restart Issues (LXC/Docker)

**Problem:** After container restart: `Unable to load libcudnn_ops.so.9`

**Solution:**
```bash
# Check if CUDA libraries are found
source venv/bin/activate
echo $LD_LIBRARY_PATH
ls -la $(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cudnn/lib/

# If empty or missing, apply the Container Restart Fix from installation section
```

**Verify fix works:**
```bash
python -c "
import torch
from faster_whisper import WhisperModel
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
model = WhisperModel('tiny', device='cuda')
print('✅ All working!')
"
```

### GPU Not Detected
```bash
nvidia-smi --query-gpu=name --format=csv,noheader,nounits
curl http://localhost:8000/health
```

### Model Loading Issues
```bash
# Validate configuration
python -c "from config_manager import WhisperConfigManager; WhisperConfigManager().validate_config()"
```

### CUDA Library Errors
```bash
# Verify LD_LIBRARY_PATH after venv activation
source venv/bin/activate
echo $LD_LIBRARY_PATH
ldconfig -p | grep cuda
```

### Blackwell GPU Performance (RTX 50XX)
If your RTX 5080/5090 is slower than expected, the system should automatically use optimized mixed precision (int8_float16). Check the startup logs for "Blackwell optimization" messages.

### Low Performance
1. **Check GPU utilization**: `nvidia-smi` during transcription
2. **Adjust batch size**: Increase in config for more VRAM
3. **Use smaller models**: `medium` or `small` for faster processing

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK**

This software is provided "as is" without warranty of any kind. The author(s) assume **NO LIABILITY** for any damages, losses, or issues that may occur from using this software, including but not limited to: data loss, hardware damage, system failures, network issues, natural disasters, cosmic events, or any other consequences whatsoever.

By using this software, you acknowledge that you use it entirely at your own risk and take full responsibility for any outcomes.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

### 🆓 Non-Commercial Use (Free)
- ✅ Personal use, education, research, open source projects

### 💼 Commercial Use License
For commercial use, open an issue in this repository or contact via GitHub.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

- OpenAI for the Whisper model
- SYSTRAN for the Faster-Whisper implementation  
- FastAPI team for the excellent web framework
