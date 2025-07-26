# Whisper Batch Transcription API

A high-performance batch transcription service using OpenAI's Whisper models with flexible configuration, optimized for GPU acceleration and designed for processing thousands of audio files efficiently.

## Features

- 🚀 **High Performance**: Support for all Whisper models (Large-v3, Large-v2, Medium, Small, etc.)
- 🌍 **Smart Language Detection**: Automatic language recognition with manual override option
- 📦 **Batch Processing**: Process up to 100 files per batch
- 📊 **Comprehensive Metrics**: Automatic performance tracking with CSV export for every batch
- 🔧 **Flexible Configuration**: Central YAML configuration for all settings
- 🔄 **Lazy Loading**: Models loaded on-demand, auto-unloaded after inactivity
- 🎵 **Real Audio Duration**: Accurate audio length measurement and real-time factor calculation
- 🔗 **Shared GPU**: Compatible with other GPU workloads (like Ollama)
- 📁 **Automatic Archiving**: All transcripts saved as numbered TXT files with metadata
- 🔧 **Runtime Configuration**: Change settings without restart via API
- 🖥️ **GPU Monitoring**: Automatic VRAM tracking and GPU utilization metrics

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (tested on RTX A2000 12GB)
- **RAM**: Minimum 16GB system RAM
- **Storage**: SSD recommended for temporary file processing
- **CUDA**: Compatible GPU with CUDA Compute Capability 6.0+

## Tested Configuration

This setup was successfully tested on:
- **Hardware**: NVIDIA RTX A2000 12GB
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **NVIDIA Driver**: 570.86.15 (with CUDA 12.8 support)
- **CUDA Toolkit**: 11.5 (pre-installed)
- **Python**: 3.10

**Note**: The system works with mixed CUDA versions - PyTorch uses CUDA 11.8 libraries while the driver supports CUDA 12.8. This is a normal and supported configuration.

## Performance & Metrics

This system doesn't just transcribe - it **automatically tracks comprehensive performance metrics** for every batch:

### 📊 **Automatic Performance Tracking**
- ⏱️ **Processing Times** - Real-time factors, throughput rates
- 🎵 **Audio Analysis** - Real audio duration measurement, format distribution  
- 🖥️ **GPU Monitoring** - VRAM usage, utilization percentages
- 🌍 **Language Detection** - Confidence scores, detection accuracy
- 📈 **Capacity Planning** - Files/hour, transcribable minutes/day projections
- 📁 **Quality Metrics** - Error rates, transcript lengths
- 💾 **All metrics saved to CSV** - `~/output/metrics.csv` for analysis

### 🔍 **View Your Performance Data**
```bash
# Real-time metrics via API
curl http://localhost:8000/metrics

# Or check the CSV file directly
cat ~/output/metrics.csv
```

### 🚀 **Performance Optimization**
The system automatically optimizes for your hardware:
- **GPU Detection** - Automatically detects and reports your GPU model
- **VRAM Monitoring** - Tracks memory usage and prevents overflows
- **Batch Size Optimization** - Configurable based on your VRAM
- **Model Selection** - Choose speed vs. quality based on your needs
- **Real-time Factors** - See exactly how much faster than real-time you're processing

### 📈 **Benchmark Your Setup**
Run your own performance tests and get detailed metrics:
- Processing speed on your specific GPU
- Optimal batch sizes for your hardware  
- Real-world throughput with your audio files
- Language detection accuracy
- Resource utilization patterns

**Every transcription run generates comprehensive performance data - perfect for optimization and capacity planning!**

## Supported Models

The system supports all Faster-Whisper models with automatic switching via configuration:

### **Large Models (Recommended):**
- `large-v3` - Latest and most accurate
- `large-v3-turbo` - Faster variant of v3
- `large-v3-german` - **Optimized for German language**
- `large-v2` - Proven and stable

### **Smaller Models:**
- `medium`, `medium.en` - Good balance of speed/quality
- `small`, `small.en` - Faster processing
- `base`, `base.en` - Basic transcription
- `tiny`, `tiny.en` - Minimal VRAM usage

## System Requirements

- **OS**: Ubuntu 22.04 LTS (tested)
- **Python**: 3.10
- **NVIDIA Driver**: 535+ with CUDA 12.0+ support
- **CUDA Toolkit**: 11.5+ (any version compatible with your driver)
- **Storage**: ~/output directory for transcripts and metrics

**Note**: The system works with mixed CUDA versions. Tested configuration: Driver 570.86.15 (CUDA 12.8), Toolkit 11.5, PyTorch CUDA 11.8 libraries.

## Installation

### Standard Installation (RTX 40XX and older)

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
git clone <your-repo-url>
cd whisper-batch-transcription

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support FIRST
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 3B. Python Environment Setup (Blackwell GPUs - RTX 50XX) 🆕

**For RTX 5090, 5080, 5070 and other Blackwell architecture GPUs:**

```bash
# Clone the repository
git clone <your-repo-url>
cd whisper-batch-transcription

# Create Python virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.4+ support for Blackwell
pip install torch==2.7.1+cu124 torchaudio==2.7.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Alternative: Use CUDA 12.1 if 12.4 has issues
# pip install torch==2.7.1+cu121 torchaudio==2.7.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies (use updated requirements for Blackwell)
pip install -r requirements-blackwell.txt
```

**Create `requirements-blackwell.txt`:**
```bash
cat > requirements-blackwell.txt << 'EOF'
# Whisper Batch Transcription API - Python Dependencies for Blackwell GPUs
# Optimized for RTX 50XX series with CUDA 12.4+

# Core ML and Audio Processing (PyTorch installed separately above)
faster-whisper==1.1.0
soundfile==0.13.1
librosa==0.11.0
numpy==2.1.2

# NVIDIA CUDA Libraries (CUDA 12.4+ compatible)
nvidia-cublas-cu12==12.6.3.3
nvidia-cudnn-cu12==9.5.1.17
nvidia-ml-py==12.575.51

# Web API Framework
fastapi==0.116.1
uvicorn==0.35.0
python-multipart==0.0.20
aiofiles==24.1.0

# Configuration and Monitoring
PyYAML==6.0.2
psutil==7.0.0
EOF
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

### 5. Verify Installation

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
GPU: NVIDIA GeForce RTX 5090  # or your specific Blackwell GPU
CUDA version: 12.4  # or higher
GPU Compute capability: (9, 0)  # Blackwell architecture
Faster-Whisper import successful!
```

### 6. Create Output Directory

```bash
# Create directory for transcripts and metrics
mkdir -p ~/output
```

## Configuration

The system uses a central YAML configuration file for all settings:

```yaml
# whisper_config.yaml
system:
  gpu_device: "cuda:0"
  max_vram_usage: 11.5  # GB (adjust based on your GPU)
  shared_gpu: true      # Allow other processes to use GPU when idle
  
single_worker:
  lazy_loading: true              # Load models on-demand
  model_timeout_minutes: 5        # Auto-unload after inactivity
  
# Main model configuration - Change here to switch between models
model:
  default_model: "large-v3-german"  # Options: "large-v2", "large-v3", "large-v3-turbo", "large-v3-german", "medium", "small", "base", "tiny"
  vram_usage_gb: 4.8               # VRAM usage (adjust for model: large=4.7GB, medium=2.4GB, small=1.2GB, base=0.8GB, tiny=0.4GB)
  batch_size: 4                    # Optimal batch size (can be higher for smaller models)
  expected_throughput: 3000        # Expected files per hour

# Language settings
language:
  default_language: "de"    # Default language (fallback if auto-detect fails)
  auto_detect: true         # Enable automatic language detection (recommended)

api:
  host: "0.0.0.0"
  port: 8000
  max_batch_size: 100
  
monitoring:
  enable_basic_logging: true
  log_level: "INFO"
```

### Model Switching

To switch models, simply change the configuration:

```yaml
# For German-optimized transcription
model:
  default_model: "large-v3-german"
  vram_usage_gb: 4.8

# For fastest processing
model:
  default_model: "medium"
  vram_usage_gb: 2.4
  batch_size: 8

# For minimal VRAM usage
model:
  default_model: "small"
  vram_usage_gb: 1.2
  batch_size: 12
```

## Usage

### Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Whisper API server
python whisper_api.py
```

The API will be available at `http://localhost:8000`

You should see:
```
🔧 Configuration loaded successfully
   Default model: large-v3-german
   Default language: de
   Max batch size: 100
🔧 SmartModelManager initialized:
   Default model: large-v3-german
   Model timeout: 5 minutes
   Batch size: 4
✅ Config validation successful
```

### API Endpoints

#### Health Check with Configuration
```bash
curl http://localhost:8000/health
```

#### Simple Batch Transcription (Auto-Detection)
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@audio1.wav" \
     -F "files=@audio2.mp3" \
     -F "files=@audio3.m4a"
# Uses config defaults: model + automatic language detection
```

#### Batch Transcription with Manual Language Override
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@english_audio.wav" \
     -F "language=en"
# Overrides auto-detection with English
```

#### Batch Transcription with Different Model
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
     -F "files=@audio.wav" \
     -F "model_name=large-v2"
# Uses large-v2 instead of config default
```

#### View Current Configuration
```bash
curl http://localhost:8000/config
```

#### View Performance Metrics
```bash
curl http://localhost:8000/metrics
```

#### Reload Configuration (Runtime Changes)
```bash
curl -X POST http://localhost:8000/config/reload
```

#### Free GPU Memory (for other workloads)
```bash
curl -X POST http://localhost:8000/free-vram
```

### Supported Audio Formats

- WAV (uncompressed)
- MP3 (compressed)
- M4A (compressed)
- OGG (compressed)
- FLAC (lossless)
- AAC (compressed)

## Output Files

### Transcripts
Transcripts are automatically saved to `~/output/transcript_XXXXXX.txt` with comprehensive metadata:

```
# Whisper Transcription #000001
# Original file: audio.wav
# Audio length: 6.25 minutes
# Processing time: 54.30s
# Real-time factor: 6.9x
# Model: large-v3-german
# Language mode: auto-detect
# Detected language: de (confidence: 0.98)
# Timestamp: 2025-07-26 14:30:15
# =====================================

[Transcript content here]
```

### Performance Metrics
Comprehensive metrics are saved to `~/output/metrics.csv` including:

- Processing times and throughput
- Real-time factors and efficiency
- GPU utilization and VRAM usage
- Language detection confidence
- Audio format analysis
- Error rates and quality metrics
- Daily/hourly capacity projections

## Language Detection

The system features intelligent language handling:

### **Automatic Detection (Default)**
```bash
curl -X POST "..." -F "files=@audio.wav"
# Whisper automatically detects language
# Works for 100+ languages
```

### **Manual Override**
```bash
curl -X POST "..." -F "files=@audio.wav" -F "language=en"
# Forces English transcription
```

### **Configuration Control**
```yaml
language:
  auto_detect: true         # Enable auto-detection
  default_language: "de"    # Fallback if detection fails
```

### **Detection Output**
```
🗣️ Language mode: auto-detect
🌍 Detected language: de (confidence: 0.98)
```

## Runtime Configuration Changes

You can modify settings without restarting the server:

### **1. Edit Configuration File**
```bash
nano whisper_config.yaml
# Change model, language settings, etc.
```

### **2. Reload via API**
```bash
curl -X POST http://localhost:8000/config/reload
```

### **3. Verify Changes**
```bash
curl http://localhost:8000/config
```

This is especially useful for:
- Switching between models for different workloads
- Adjusting batch sizes based on file types
- Changing timeout settings during heavy usage

## Docker Support (Optional)

For containerized deployment:

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Set up application
WORKDIR /app
COPY requirements.txt whisper_config.yaml .
RUN python3.10 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["./venv/bin/python", "whisper_api.py"]
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/whisper-batch.service`:

```ini
[Unit]
Description=Whisper Batch Transcription API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/whisper-batch-transcription
Environment=PATH=/path/to/whisper-batch-transcription/venv/bin
ExecStart=/path/to/whisper-batch-transcription/venv/bin/python whisper_api.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable whisper-batch.service
sudo systemctl start whisper-batch.service
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 500M;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Troubleshooting

### Common Issues

**GPU Not Detected in Metrics:**
```bash
# Test GPU detection
nvidia-smi --query-gpu=name --format=csv,noheader,nounits | sed 's/^NVIDIA //'

# Check API health
curl http://localhost:8000/health
```

**Model Loading Issues:**
```bash
# Test model support
curl http://localhost:8000/config

# Validate configuration
python -c "from config_manager import WhisperConfigManager; WhisperConfigManager().validate_config()"
```

**CUDA/cuDNN Library Errors:**
```bash
# Verify LD_LIBRARY_PATH is set correctly
echo $LD_LIBRARY_PATH

# Check if CUDA libraries are found
ldconfig -p | grep cuda
```

### Blackwell GPU Specific Issues 🆕

**CUDA Compatibility Issues (RTX 50XX):**
```bash
# Check CUDA version compatibility
nvidia-smi
nvcc --version

# Verify PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# For Blackwell, you need CUDA 12.4+ and matching PyTorch
```

**Compute Capability Not Supported:**
```bash
# Check if your GPU compute capability is supported
python -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap}')
    if cap[0] >= 9:  # Blackwell is 9.0
        print('✅ Blackwell GPU detected and supported')
    else:
        print(f'⚠️ Older GPU architecture: {cap}')
else:
    print('❌ CUDA not available')
"
```

**Memory Issues on High-End Blackwell GPUs:**
```bash
# RTX 5090 has 32GB VRAM - you can run larger models!
# Update config for better performance:

# In whisper_config.yaml for RTX 5090:
model:
  default_model: "large-v3"
  vram_usage_gb: 4.7
  batch_size: 8  # Increase batch size for more VRAM
```

**faster-whisper Compatibility:**
```bash
# If you get model loading errors, try newest faster-whisper
pip install --upgrade faster-whisper

# Or specific version known to work with Blackwell:
pip install faster-whisper>=1.1.0
```

**PyTorch CUDA Version Mismatch:**
```bash
# If you get CUDA runtime errors:
pip uninstall torch torchaudio
pip install torch==2.7.1+cu124 torchaudio==2.7.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Verify installation
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
"
```

**Driver Version Requirements:**
```bash
# Blackwell GPUs need driver 560+ 
nvidia-smi | head -3

# If driver is too old:
sudo apt update
sudo apt install nvidia-driver-560  # or latest
sudo reboot
```

### Language Detection Issues

```bash
# Check configuration
curl http://localhost:8000/config | grep auto_detect

# Test with manual language
curl -X POST "..." -F "files=@audio.wav" -F "language=de"
```

### GPU Memory Issues

```bash
# Check GPU memory usage
nvidia-smi

# Free GPU memory via API
curl -X POST http://localhost:8000/free-vram

# Adjust model in config for lower VRAM usage
# large-v3: 4.7GB → medium: 2.4GB → small: 1.2GB
```

**Configuration Validation Errors:**
```bash
# Check config syntax
python -c "import yaml; yaml.safe_load(open('whisper_config.yaml'))"

# View detailed validation
python whisper_api.py
```

### Performance Optimization

1. **Model Selection**: Choose appropriate model for quality/speed trade-off
   - `large-v3-german` for best German accuracy
   - `medium` for balanced performance
   - `small` for maximum speed

2. **VRAM Optimization**: 
   - Use smaller models if memory is limited
   - Adjust `batch_size` in config
   - Monitor usage with `nvidia-smi`

3. **Storage**: Use SSD for temporary files

4. **Language Settings**: 
   - Disable auto-detection if you know the language
   - Use language-specific models when available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

### 🆓 Non-Commercial Use (Free)

You are free to:
- ✅ **Share** — copy and redistribute in any medium or format
- ✅ **Adapt** — remix, transform, and build upon the material
- ✅ **Personal use** — individual transcription projects
- ✅ **Educational use** — research, learning, teaching
- ✅ **Open source projects** — non-profit community projects

**Under the following terms:**
- **Attribution** — You must give appropriate credit and link to the license
- **NonCommercial** — You may not use the material for commercial purposes

### 💼 Commercial Use License

For commercial use, you need a separate commercial license:
- 🏢 **Business services** — Offering transcription services to customers
- 💰 **Revenue generation** — Any use that generates income  
- 🔗 **SaaS platforms** — Commercial web services
- 🏭 **Enterprise deployment** — Large-scale commercial use

**To obtain a commercial license:**
- Email: [your.email@domain.de]
- Subject: "Commercial License - Whisper Batch API"
- Describe your intended commercial use case

---

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Full license text:** [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

- OpenAI for the Whisper model
- SYSTRAN for the Faster-Whisper implementation
- FastAPI team for the excellent web framework
