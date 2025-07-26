from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import os
import time
import subprocess  # Add this import for nvidia-smi command
from smart_model_manager import SmartModelManager
from config_manager import WhisperConfigManager
import uvicorn
from typing import List
import asyncio
from pathlib import Path
import csv
from datetime import datetime
import pynvml
from collections import Counter
import soundfile as sf  # For real audio length measurement

# FastAPI App
app = FastAPI(title="Whisper Large-v2/v3 Batch API", version="1.0.0")

# Load Configuration
try:
    config = WhisperConfigManager("whisper_config.yaml")
    print(f"🔧 Configuration loaded successfully")
    print(f"   Default model: {config.get_default_model()}")
    print(f"   Default language: {config.get_default_language()}")
    print(f"   Max batch size: {config.get_max_batch_size()}")
except Exception as e:
    print(f"❌ Failed to load configuration: {e}")
    raise RuntimeError("Cannot start without valid configuration")

# Global Model Manager (uses config)
model_manager = SmartModelManager("whisper_config.yaml")

# Use output directory from config instead of hardcoded ~/output
OUTPUT_DIR = Path(config.get_output_directory())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics CSV
METRICS_CSV = OUTPUT_DIR / "metrics.csv"

def get_real_audio_duration(file_path):
    """Measure real audio duration in minutes"""
    try:
        # Try with soundfile
        info = sf.info(file_path)
        duration_seconds = info.duration
        return duration_seconds / 60  # Minutes
    except Exception as e1:
        try:
            # Fallback: librosa
            import librosa
            duration_seconds = librosa.get_duration(path=file_path)
            return duration_seconds / 60  # Minutes
        except Exception as e2:
            print(f"⚠️ Could not measure audio duration: {e1}, {e2}")
            return 0

def init_metrics_csv():
    """Create CSV header if file doesn't exist"""
    if not METRICS_CSV.exists():
        headers = [
            'index', 'date', 'time', 'model', 'gpu_model', 'batch_size', 'total_files',
            'successful_files', 'failed_files', 'avg_file_size_mb', 'avg_audio_length_min',
            'avg_processing_time_s', 'total_batch_time_s', 'files_per_hour', 'files_per_day',
            'audio_minutes_per_hour', 'audio_minutes_per_day', 'throughput_mb_per_s',
            'vram_used_mb', 'gpu_utilization_percent', 'error_rate_percent',
            'avg_language_confidence', 'audio_formats', 'model_load_time_s',
            'avg_transcript_length_chars', 'transcribable_minutes_per_hour',
            'transcribable_minutes_per_day', 'real_time_factor'
        ]
        
        with open(METRICS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def get_next_file_number():
    """Determine next available file number"""
    existing_files = list(OUTPUT_DIR.glob("transcript_*.txt"))
    if not existing_files:
        return 1
    
    numbers = []
    for file in existing_files:
        try:
            num = int(file.stem.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def get_next_metrics_index():
    """Next metrics index number"""
    if not METRICS_CSV.exists():
        return 1
    
    try:
        with open(METRICS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return len(rows) + 1 if rows else 1
    except:
        return 1

def save_metrics_to_csv(metrics_data):
    """Add metrics to CSV"""
    init_metrics_csv()
    
    with open(METRICS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_data.keys())
        writer.writerow(metrics_data)

def get_gpu_info():
    """Gather GPU information using nvidia-smi command"""
    try:
        # Get GPU name using nvidia-smi command (more reliable)
        gpu_name_cmd = "nvidia-smi --query-gpu=name --format=csv,noheader,nounits | sed 's/^NVIDIA //'"
        gpu_name_result = subprocess.run(gpu_name_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if gpu_name_result.returncode == 0 and gpu_name_result.stdout.strip():
            gpu_name = gpu_name_result.stdout.strip()
        else:
            gpu_name = 'Unknown'
        
        # Try to use pynvml for memory and utilization info
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            vram_used_mb = memory_info.used / (1024**2)
            gpu_utilization = util.gpu
        except Exception as pynvml_error:
            print(f"⚠️ pynvml error (using fallback): {pynvml_error}")
            # Fallback: get memory info via nvidia-smi
            try:
                mem_cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
                mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True, timeout=10)
                vram_used_mb = float(mem_result.stdout.strip()) if mem_result.returncode == 0 else 0
                
                util_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                util_result = subprocess.run(util_cmd, shell=True, capture_output=True, text=True, timeout=10)
                gpu_utilization = int(util_result.stdout.strip()) if util_result.returncode == 0 else 0
            except:
                vram_used_mb = 0
                gpu_utilization = 0
        
        print(f"🔍 GPU Info - Name: {gpu_name}, VRAM: {vram_used_mb:.0f}MB, Utilization: {gpu_utilization}%")
        
        return {
            'gpu_model': gpu_name,
            'vram_used_mb': vram_used_mb,
            'gpu_utilization_percent': gpu_utilization
        }
        
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")
        # Last resort fallback
        try:
            gpu_name_cmd = "nvidia-smi --query-gpu=name --format=csv,noheader,nounits | sed 's/^NVIDIA //'"
            gpu_name_result = subprocess.run(gpu_name_cmd, shell=True, capture_output=True, text=True, timeout=5)
            gpu_name = gpu_name_result.stdout.strip() if gpu_name_result.returncode == 0 else 'Unknown'
        except:
            gpu_name = 'Unknown'
            
        return {
            'gpu_model': gpu_name,
            'vram_used_mb': 0,
            'gpu_utilization_percent': 0
        }

def save_transcript_to_file(transcript, filename, processing_time, model_used, audio_length_min=0, detected_language="unknown", language_confidence=0.0, language_mode="unknown"):
    """Save transcript as TXT file"""
    file_number = get_next_file_number()
    output_file = OUTPUT_DIR / f"transcript_{file_number:06d}.txt"
    
    content = f"""# Whisper Transcription #{file_number:06d}
# Original file: {filename}
# Audio length: {audio_length_min:.2f} minutes
# Processing time: {processing_time:.2f}s
# Real-time factor: {(audio_length_min * 60 / processing_time):.1f}x
# Model: {model_used}
# Language mode: {language_mode}
# Detected language: {detected_language} (confidence: {language_confidence:.2f})
# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
# =====================================

{transcript}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Transcript saved: {output_file}")
    return output_file

@app.get("/health")
async def health_check():
    """API Health Check with configuration info"""
    gpu_info = get_gpu_info()
    transcript_count = len(list(OUTPUT_DIR.glob("transcript_*.txt")))
    model_status = model_manager.get_model_status()
    
    return {
        "status": "healthy",
        "gpu_name": gpu_info['gpu_model'],
        "vram_used": f"{gpu_info['vram_used_mb']:.0f}MB",
        "gpu_utilization": f"{gpu_info['gpu_utilization_percent']}%",
        "configuration": {
            "default_model": config.get_default_model(),
            "default_language": config.get_default_language(),
            "auto_detect_enabled": config.is_auto_detect_enabled(),
            "max_batch_size": config.get_max_batch_size(),
            "model_timeout_minutes": config.get_model_timeout_minutes(),
            "batch_size": config.get_model_batch_size(),
            "supported_models": model_manager.get_supported_models(),
            "output_directory": config.get_output_directory()
        },
        "model_status": model_status,
        "shared_gpu": True,
        "worker_type": "single_optimized_with_config",
        "transcripts_saved": transcript_count,
        "output_directory": str(OUTPUT_DIR.absolute()),
        "metrics_csv": str(METRICS_CSV.absolute())
    }

@app.post("/transcribe/batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form(None),  # Will use config default if None
    language: str = Form(None)     # Will use auto-detection or config default if None
):
    """Batch transcription with real audio length measurement and config defaults"""
    
    # Use config default model if not specified
    if model_name is None:
        model_name = config.get_default_model()
    
    # Handle language detection logic
    whisper_language = None  # For Whisper API call
    language_mode = "unknown"
    
    if language is not None:
        # User specified a language - use it directly
        whisper_language = language
        language_mode = f"manual:{language}"
    elif config.is_auto_detect_enabled():
        # Auto-detection enabled and no language specified - let Whisper auto-detect
        whisper_language = None
        language_mode = "auto-detect"
    else:
        # Auto-detection disabled - use config default
        whisper_language = config.get_default_language()
        language_mode = f"default:{whisper_language}"
    
    # Validate model
    if not config.is_model_supported(model_name):
        raise HTTPException(400, f"Unsupported model: {model_name}. Supported: {model_manager.get_supported_models()}")
    
    # Check batch size limit
    if len(files) > config.get_max_batch_size():
        raise HTTPException(400, f"Maximum {config.get_max_batch_size()} files per batch")
    
    print(f"🚀 Batch received: {len(files)} files")
    print(f"📋 Configuration: Model={model_name}, Language={language_mode}")
    print(f"📝 File list: {[f.filename for f in files]}")
    
    # Start metric collection
    model_load_start = time.time()
    
    # Load model in advance
    model_instance = model_manager.get_model(model_name)
    model_load_time = time.time() - model_load_start
    
    results = []
    total_start_time = time.time()
    saved_files = []
    
    # Collection for metrics
    file_sizes = []
    processing_times = []
    transcript_lengths = []
    language_confidences = []
    audio_formats = []
    real_audio_lengths = []
    
    for i, file in enumerate(files):
        print(f"\n--- Processing {i+1}/{len(files)}: {file.filename} ---")
        
        file_path = f"/tmp/whisper_batch_{int(time.time())}_{i}_{file.filename}"
        
        try:
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            file_size_mb = len(content) / (1024**2)
            file_sizes.append(file_size_mb)
            
            # Audio format
            file_ext = os.path.splitext(file.filename)[1]
            audio_formats.append(file_ext)
            
            # Measure REAL audio length
            real_audio_length = get_real_audio_duration(file_path)
            real_audio_lengths.append(real_audio_length)
            
            print(f"📊 File size: {file_size_mb:.1f}MB")
            print(f"🎵 REAL audio length: {real_audio_length:.2f} minutes")
            
            # Transcription
            print(f"🎯 Starting transcription for {file.filename}")
            print(f"🗣️ Language mode: {language_mode}")
            start_time = time.time()
            
            segments, info = model_instance['model'].transcribe(
                file_path,
                language=whisper_language,  # None for auto-detect, or specific language
                task="transcribe"
            )
            
            transcript = " ".join([segment.text for segment in segments])
            processing_time = time.time() - start_time
            
            # Calculate real-time factor
            real_time_factor = (real_audio_length * 60 / processing_time) if processing_time > 0 else 0
            
            processing_times.append(processing_time)
            transcript_lengths.append(len(transcript))
            language_confidences.append(info.language_probability)
            
            print(f"⏱️ Transcription completed in {processing_time:.2f}s")
            print(f"🌍 Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            print(f"⚡ Real-time factor: {real_time_factor:.1f}x faster")
            print(f"📄 Transcript length: {len(transcript)} characters")
            
            # Save transcript (with real audio length and language info)
            output_file = save_transcript_to_file(
                transcript, 
                file.filename, 
                processing_time, 
                model_name, 
                real_audio_length,
                info.language,
                info.language_probability,
                language_mode
            )
            saved_files.append(str(output_file))
            
            result = {
                "filename": file.filename,
                "transcript": transcript.strip(),
                "detected_language": info.language,
                "language_probability": info.language_probability,
                "processing_time": processing_time,
                "real_audio_length_min": real_audio_length,
                "real_time_factor": real_time_factor,
                "status": "success",
                "saved_to": str(output_file),
                "file_size_mb": file_size_mb
            }
            
            results.append(result)
            print(f"✅ Result #{len(results)} added for {file.filename}")
            
        except Exception as e:
            print(f"❌ Error with {file.filename}: {str(e)}")
            error_result = {
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            }
            results.append(error_result)
        
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    total_time = time.time() - total_start_time
    successful = len([r for r in results if r["status"] == "success"])
    failed = len(files) - successful
    
    # Calculate metrics
    if successful > 0 and real_audio_lengths:
        avg_file_size = sum(file_sizes) / len(file_sizes)
        avg_processing_time = sum(processing_times) / len(processing_times)
        avg_transcript_length = sum(transcript_lengths) / len(transcript_lengths)
        avg_language_confidence = sum(language_confidences) / len(language_confidences)
        avg_audio_length = sum(real_audio_lengths) / len(real_audio_lengths)
        
        files_per_hour = 3600 / avg_processing_time if avg_processing_time > 0 else 0
        files_per_day = files_per_hour * 24
        
        total_data_mb = sum(file_sizes)
        throughput_mb_per_s = total_data_mb / total_time if total_time > 0 else 0
        
        audio_minutes_per_hour = files_per_hour * avg_audio_length
        audio_minutes_per_day = audio_minutes_per_hour * 24
        
        # Real-time factor
        real_time_factor = (avg_audio_length * 60 / avg_processing_time) if avg_processing_time > 0 else 0
        transcribable_minutes_per_hour = 60 * real_time_factor
        transcribable_minutes_per_day = transcribable_minutes_per_hour * 24
        
        error_rate = (failed / len(files)) * 100
        gpu_info = get_gpu_info()
        
        # Format distribution
        format_counter = Counter(audio_formats)
        formats_str = ",".join([f"{ext}:{count}" for ext, count in format_counter.items()])
        
        # Save metrics
        metrics_data = {
            'index': get_next_metrics_index(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'model': model_name,
            'gpu_model': gpu_info['gpu_model'],
            'batch_size': len(files),
            'total_files': len(files),
            'successful_files': successful,
            'failed_files': failed,
            'avg_file_size_mb': round(avg_file_size, 2),
            'avg_audio_length_min': round(avg_audio_length, 2),
            'avg_processing_time_s': round(avg_processing_time, 2),
            'total_batch_time_s': round(total_time, 2),
            'files_per_hour': round(files_per_hour, 1),
            'files_per_day': round(files_per_day, 0),
            'audio_minutes_per_hour': round(audio_minutes_per_hour, 1),
            'audio_minutes_per_day': round(audio_minutes_per_day, 0),
            'throughput_mb_per_s': round(throughput_mb_per_s, 2),
            'vram_used_mb': round(gpu_info['vram_used_mb'], 0),
            'gpu_utilization_percent': gpu_info['gpu_utilization_percent'],
            'error_rate_percent': round(error_rate, 1),
            'avg_language_confidence': round(avg_language_confidence, 3),
            'audio_formats': formats_str,
            'model_load_time_s': round(model_load_time, 2),
            'avg_transcript_length_chars': round(avg_transcript_length, 0),
            'transcribable_minutes_per_hour': round(transcribable_minutes_per_hour, 1),
            'transcribable_minutes_per_day': round(transcribable_minutes_per_day, 0),
            'real_time_factor': round(real_time_factor, 1)
        }
        
        save_metrics_to_csv(metrics_data)
        
        print(f"\n📊 Real performance metrics:")
        print(f"   GPU: {gpu_info['gpu_model']}")
        print(f"   Average audio length: {avg_audio_length:.2f} min")
        print(f"   Real-time factor: {real_time_factor:.1f}x")
        print(f"   Transcribable minutes/hour: {transcribable_minutes_per_hour:.1f}")
        print(f"   Files/hour: {files_per_hour:.1f}")
    
    print(f"\n🏁 Batch completed:")
    print(f"   Results list length: {len(results)}")
    print(f"   Successful files: {successful}")
    print(f"   Saved TXT files: {len(saved_files)}")
    
    return {
        "batch_summary": {
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "avg_time_per_file": total_time / len(files),
            "model_used": model_name,
            "files_saved": len(saved_files),
            "output_directory": str(OUTPUT_DIR.absolute())
        },
        "results": results,
        "saved_files": saved_files
    }

@app.get("/metrics")
async def get_metrics():
    """Load all metrics from CSV"""
    if not METRICS_CSV.exists():
        return {"metrics": [], "total_entries": 0}
    
    with open(METRICS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        metrics = list(reader)
    
    return {
        "metrics": metrics,
        "total_entries": len(metrics),
        "csv_file": str(METRICS_CSV.absolute())
    }

@app.post("/free-vram")
async def free_vram():
    """Free VRAM for other containers"""
    model_manager.force_unload_all()
    return {"status": "vram_freed", "message": "VRAM freed for Ollama/other containers"}

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "configuration": {
            "default_model": config.get_default_model(),
            "default_language": config.get_default_language(),
            "auto_detect_enabled": config.is_auto_detect_enabled(),
            "max_batch_size": config.get_max_batch_size(),
            "model_timeout_minutes": config.get_model_timeout_minutes(),
            "batch_size": config.get_model_batch_size(),
            "supported_models": model_manager.get_supported_models(),
            "api_host_port": config.get_api_host_port(),
            "output_directory": config.get_output_directory()
        },
        "model_status": model_manager.get_model_status()
    }

@app.post("/config/reload")
async def reload_config():
    """Reload configuration from file (useful for runtime changes)"""
    try:
        config.reload_config()
        model_manager.reload_config()
        return {
            "status": "config_reloaded",
            "new_config": {
                "default_model": config.get_default_model(),
                "default_language": config.get_default_language(),
                "model_timeout_minutes": config.get_model_timeout_minutes(),
                "output_directory": config.get_output_directory()
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to reload config: {str(e)}")

if __name__ == "__main__":
    host, port = config.get_api_host_port()
    print(f"🚀 Starting Whisper API server on {host}:{port}")
    print(f"🔧 Using model: {config.get_default_model()}")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    uvicorn.run(app, host=host, port=port)
