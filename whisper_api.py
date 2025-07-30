from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import os
import time
import threading
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
import concurrent.futures
from dataclasses import dataclass

# FastAPI App
app = FastAPI(title="Whisper Large-v2/v3 Batch API", version="1.0.0")

@dataclass
class FileProcessingTask:
    """Data structure for individual file processing tasks"""
    file: UploadFile
    index: int
    file_path: str
    file_size_mb: float
    real_audio_length: float
    audio_format: str

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
            'transcribable_minutes_per_day', 'real_time_factor', 'concurrent_processing',
            'max_concurrent_files'
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

def get_optimal_gpu_settings(gpu_name, total_vram_gb, base_model_vram=4.7, overhead_per_file=0.3, safety_buffer=0.5, avg_file_length_min=6.0):
    """Pure VRAM-based scaling that works for ANY GPU model"""
    
    # Use configurable parameters
    BASE_MODEL_VRAM = base_model_vram
    BASE_OVERHEAD = overhead_per_file
    VRAM_SAFETY_BUFFER = safety_buffer
    
    # Calculate available VRAM for concurrent processing
    usable_vram = total_vram_gb - BASE_MODEL_VRAM - VRAM_SAFETY_BUFFER
    
    if usable_vram <= 0:
        print(f"⚠️ Very low VRAM ({total_vram_gb:.1f}GB) - using minimal settings")
        return {'max_concurrent': 1, 'batch_size_multiplier': 0.5, 'vram_usage': 0.6}
    
    # DYNAMIC FILE LENGTH SCALING: More aggressive for short files
    # Shorter files need less VRAM per file, longer files need more
    length_factor = max(0.3, min(2.5, avg_file_length_min / 10))  # 0.3x to 2.5x scaling (more aggressive)
    adjusted_overhead = BASE_OVERHEAD * length_factor
    
    # Calculate max concurrent files based on adjusted VRAM usage
    max_concurrent_by_vram = int(usable_vram / adjusted_overhead)
    
    # ULTRA AGGRESSIVE scaling for maximum VRAM utilization
    if total_vram_gb >= 80:  # Ultra high-end (H100, B200, etc.)
        vram_usage = 0.95
        batch_multiplier = 1.5
        max_concurrent = min(max_concurrent_by_vram, 100)  # Even more aggressive
    elif total_vram_gb >= 40:  # High-end (A100, A6000 Ada, etc.)
        vram_usage = 0.93
        batch_multiplier = 1.3
        max_concurrent = min(max_concurrent_by_vram, 60)   # Even more aggressive
    elif total_vram_gb >= 20:  # Performance (RTX 4090, etc.)
        vram_usage = 0.90
        batch_multiplier = 1.2
        max_concurrent = min(max_concurrent_by_vram, 45)   # Even more aggressive
    elif total_vram_gb >= 16:  # Upper mid-range (RTX 5080, etc.)
        vram_usage = 0.88
        batch_multiplier = 1.1
        max_concurrent = min(max_concurrent_by_vram, 35)   # Even more aggressive
    elif total_vram_gb >= 12:  # Mid-range (A2000, etc.)
        vram_usage = 0.85
        batch_multiplier = 0.9
        max_concurrent = min(max_concurrent_by_vram, 25)   # Even more aggressive
    elif total_vram_gb >= 8:   # Entry-level
        vram_usage = 0.82
        batch_multiplier = 0.7
        max_concurrent = min(max_concurrent_by_vram, 15)   # Even more aggressive
    elif total_vram_gb >= 4:   # Very low-end
        vram_usage = 0.80
        batch_multiplier = 0.6
        max_concurrent = min(max_concurrent_by_vram, 8)    # Even more aggressive
    else:  # <4GB - probably won't work but try
        vram_usage = 0.65
        batch_multiplier = 0.5
        max_concurrent = 1
    
    # Ensure minimum of 1 concurrent file
    max_concurrent = max(1, max_concurrent)
    
    print(f"🚀 DYNAMIC scaling: {total_vram_gb:.1f}GB → {max_concurrent} concurrent files")
    print(f"   File length: {avg_file_length_min:.1f}min → {length_factor:.1f}x overhead factor")
    print(f"   Base overhead: {BASE_OVERHEAD:.2f}GB → Adjusted: {adjusted_overhead:.2f}GB per file")
    print(f"   VRAM allows: {max_concurrent_by_vram}, Using: {max_concurrent}")
    print(f"   VRAM efficiency: {(BASE_MODEL_VRAM + max_concurrent * adjusted_overhead) / total_vram_gb:.0%}")
    
    return {
        'max_concurrent': max_concurrent,
        'batch_size_multiplier': batch_multiplier,
        'vram_usage': vram_usage
    }

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
            vram_total_gb = memory_info.total / (1024**3)  # Get total VRAM in GB
            gpu_utilization = util.gpu
        except Exception as pynvml_error:
            print(f"⚠️ pynvml error (using fallback): {pynvml_error}")
            # Fallback: get memory info via nvidia-smi
            try:
                mem_cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
                mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True, timeout=10)
                vram_used_mb = float(mem_result.stdout.strip()) if mem_result.returncode == 0 else 0
                
                # Get total VRAM
                total_mem_cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
                total_result = subprocess.run(total_mem_cmd, shell=True, capture_output=True, text=True, timeout=10)
                vram_total_gb = float(total_result.stdout.strip()) / 1024 if total_result.returncode == 0 else 12  # Default 12GB
                
                util_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
                util_result = subprocess.run(util_cmd, shell=True, capture_output=True, text=True, timeout=10)
                gpu_utilization = int(util_result.stdout.strip()) if util_result.returncode == 0 else 0
            except:
                vram_used_mb = 0
                vram_total_gb = 12  # Default fallback
                gpu_utilization = 0
        
        print(f"🔍 GPU Info - Name: {gpu_name}, VRAM: {vram_used_mb:.0f}MB/{vram_total_gb:.1f}GB, Utilization: {gpu_utilization}%")
        
        return {
            'gpu_model': gpu_name,
            'vram_used_mb': vram_used_mb,
            'vram_total_gb': vram_total_gb,
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
            'vram_total_gb': 12,  # Default fallback
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

def process_single_file_sync(task: FileProcessingTask, model_instance, whisper_language, language_mode, model_name):
    """Synchronous function to process a single file (for use with ThreadPoolExecutor)"""
    try:
        print(f"🎯 Processing file {task.index + 1}: {task.file.filename}")
        print(f"🎵 Audio length: {task.real_audio_length:.2f} min, Size: {task.file_size_mb:.1f}MB")
        
        start_time = time.time()
        batch_size = model_instance['batch_size']
        
        # Try with full batch size first
        try:
            segments, info = model_instance['model'].transcribe(
                task.file_path,
                language=whisper_language,
                task="transcribe",
                batch_size=batch_size
            )
        except Exception as e:
            # If CUDA OOM, try with reduced batch size
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"⚠️ CUDA OOM detected, reducing batch size from {batch_size} to {batch_size//2}")
                segments, info = model_instance['model'].transcribe(
                    task.file_path,
                    language=whisper_language,
                    task="transcribe",
                    batch_size=max(1, batch_size//2)
                )
            else:
                raise e
        
        transcript = " ".join([segment.text for segment in segments])
        processing_time = time.time() - start_time
        real_time_factor = (task.real_audio_length * 60 / processing_time) if processing_time > 0 else 0
        
        print(f"⏱️ File {task.index + 1} completed in {processing_time:.2f}s (RTF: {real_time_factor:.1f}x)")
        
        # Save transcript
        output_file = save_transcript_to_file(
            transcript, 
            task.file.filename, 
            processing_time, 
            model_name, 
            task.real_audio_length,
            info.language,
            info.language_probability,
            language_mode
        )
        
        return {
            "filename": task.file.filename,
            "transcript": transcript.strip(),
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "processing_time": processing_time,
            "real_audio_length_min": task.real_audio_length,
            "real_time_factor": real_time_factor,
            "status": "success",
            "saved_to": str(output_file),
            "file_size_mb": task.file_size_mb,
            "transcript_length": len(transcript)
        }
        
    except Exception as e:
        print(f"❌ Error processing {task.file.filename}: {str(e)}")
        return {
            "filename": task.file.filename,
            "error": str(e),
            "status": "failed"
        }

def check_vram_safety(max_concurrent, gpu_info, estimated_model_vram_gb=5.0):
    """Check if concurrent processing is safe for VRAM"""
    total_vram_gb = gpu_info['vram_total_gb']
    used_vram_gb = gpu_info['vram_used_mb'] / 1024
    available_vram_gb = total_vram_gb - used_vram_gb
    
    # Estimate VRAM needed (model + overhead per concurrent process)
    estimated_needed_gb = estimated_model_vram_gb + (max_concurrent * 0.5)  # 0.5GB overhead per concurrent file
    
    if estimated_needed_gb > available_vram_gb * 0.95:  # 95% safety threshold
        # Reduce concurrency to fit in available VRAM
        safe_concurrent = max(1, int(available_vram_gb / (estimated_model_vram_gb / max_concurrent + 0.5)))
        print(f"⚠️ VRAM Safety: Reducing from {max_concurrent} to {safe_concurrent} concurrent files")
        print(f"   Available VRAM: {available_vram_gb:.1f}GB, Estimated need: {estimated_needed_gb:.1f}GB")
        return safe_concurrent
    
    return max_concurrent

class DynamicVRAMScaler:
    """Real-time VRAM monitoring and concurrent file scaling"""
    
    def __init__(self, config, initial_concurrent):
        self.config = config
        self.current_concurrent = initial_concurrent
        self.files_processed = 0
        self.scaling_enabled = config.concurrent_processing.dynamic_vram_scaling
        self.target_min = config.concurrent_processing.target_vram_min
        self.target_max = config.concurrent_processing.target_vram_max
        self.step_size = config.concurrent_processing.scaling_step_size
        self.check_interval = config.concurrent_processing.scaling_check_interval
        self.time_interval = config.concurrent_processing.scaling_time_interval
        self.last_scaling_time = time.time()
        
    def should_check_scaling(self):
        """Check if we should evaluate VRAM and adjust scaling"""
        current_time = time.time()
        time_based_check = (current_time - self.last_scaling_time) >= self.time_interval
        file_based_check = (self.files_processed > 0 and 
                           self.files_processed % self.check_interval == 0)
        
        return (self.scaling_enabled and (time_based_check or file_based_check))
    
    def get_current_vram_usage(self):
        """Get current VRAM usage percentage"""
        try:
            gpu_info = get_gpu_info()
            return gpu_info['vram_used_mb'] / (gpu_info['vram_total_gb'] * 1024)
        except:
            return 0.5  # Fallback
    
    def adjust_concurrency(self):
        """Dynamically adjust concurrent files based on VRAM usage"""
        try:
            # Get fresh GPU info for accurate VRAM measurement
            gpu_info = get_gpu_info()
            current_vram_pct = self.get_current_vram_usage()
            
            # Enhanced logging with actual VRAM values
            vram_used_gb = gpu_info['vram_used_mb'] / 1024
            vram_total_gb = gpu_info['vram_total_gb']
            print(f"🔍 GPU Info - Name: {gpu_info['gpu_model']}, VRAM: {gpu_info['vram_used_mb']:.0f}MB/{vram_total_gb:.1f}GB, Utilization: {gpu_info['gpu_utilization_percent']}%")
            
            if current_vram_pct < self.target_min:
                # VRAM usage too low - increase concurrency
                old_concurrent = self.current_concurrent
                
                # Dynamic max concurrent limit based on GPU VRAM (more aggressive now)
                max_safe_concurrent = max(6, int(vram_total_gb * 3))  # ~3 files per GB for maximum utilization
                self.current_concurrent = min(self.current_concurrent + self.step_size, max_safe_concurrent)
                
                print(f"🔼 VRAM {current_vram_pct:.0%} < {self.target_min:.0%} → Increasing {old_concurrent} → {self.current_concurrent} concurrent files (max: {max_safe_concurrent})")
                self.last_scaling_time = time.time()  # Reset time counter
                return self.current_concurrent
                
            elif current_vram_pct > self.target_max:
                # VRAM usage too high - decrease concurrency  
                old_concurrent = self.current_concurrent
                self.current_concurrent = max(self.current_concurrent - self.step_size, 1)  # Min 1
                print(f"🔽 VRAM {current_vram_pct:.0%} > {self.target_max:.0%} → Decreasing {old_concurrent} → {self.current_concurrent} concurrent files")
                self.last_scaling_time = time.time()  # Reset time counter
                return self.current_concurrent
            else:
                # VRAM usage in target range
                print(f"✅ VRAM {current_vram_pct:.0%} in target range {self.target_min:.0%}-{self.target_max:.0%} → Keeping {self.current_concurrent} concurrent files")
                return self.current_concurrent
                
        except Exception as e:
            print(f"⚠️ Error getting GPU info for dynamic scaling: {e}")
            return self.current_concurrent
    
    def file_completed(self):
        """Call this when a file is completed"""
        self.files_processed += 1
        
        if self.should_check_scaling():
            return self.adjust_concurrency()
        return self.current_concurrent

async def process_files_concurrently(files: List[UploadFile], model_instance, whisper_language, language_mode, model_name):
    """Process files concurrently with dynamic VRAM scaling"""
    
    # Prepare all files first
    tasks = []
    for i, file in enumerate(files):
        file_path = f"/tmp/whisper_batch_{int(time.time())}_{i}_{file.filename}"
        
        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size_mb = len(content) / (1024**2)
        file_ext = os.path.splitext(file.filename)[1]
        real_audio_length = get_real_audio_duration(file_path)
        
        task = FileProcessingTask(
            file=file,
            index=i,
            file_path=file_path,
            file_size_mb=file_size_mb,
            real_audio_length=real_audio_length,
            audio_format=file_ext
        )
        tasks.append(task)
    
    # Determine optimal concurrency based on config, GPU, and file characteristics
    max_concurrent = config.get_max_concurrent_files()
    
    # Auto GPU scaling if enabled
    if config.is_auto_gpu_scaling_enabled():
        # Get current GPU info
        gpu_info = get_gpu_info()
        
        # Calculate average file length for dynamic scaling
        avg_file_length = sum(task.real_audio_length for task in tasks) / len(tasks) if tasks else 6.0
        
        gpu_settings = get_optimal_gpu_settings(
            gpu_info['gpu_model'], 
            gpu_info['vram_total_gb'],
            config.concurrent_processing.base_model_vram_gb,
            config.concurrent_processing.overhead_per_file_gb,
            config.concurrent_processing.safety_buffer_gb,
            avg_file_length
        )
        max_concurrent = gpu_settings['max_concurrent']
        print(f"🎯 Auto GPU scaling: {gpu_info['gpu_model']} -> max_concurrent={max_concurrent}")
        
        # VRAM safety check to prevent crashes
        max_concurrent = check_vram_safety(max_concurrent, gpu_info)
    
    if config.is_adaptive_batching_enabled():
        # Adjust based on file sizes and estimated VRAM usage
        avg_file_duration = sum(task.real_audio_length for task in tasks) / len(tasks)
        if avg_file_duration < 3.0:  # Very short files
            max_concurrent = min(max_concurrent + 1, max_concurrent * 1.3)  # Allow more concurrency
        elif avg_file_duration > 10.0:  # Longer files
            max_concurrent = max(max_concurrent - 1, 2)  # Reduce concurrency
        
        max_concurrent = int(max_concurrent)  # Ensure integer
    
    # Initialize dynamic VRAM scaler
    vram_scaler = DynamicVRAMScaler(config, max_concurrent)
    current_concurrent = max_concurrent
    
    print(f"🚀 Processing {len(tasks)} files with initial max_concurrent={max_concurrent}")
    if vram_scaler.scaling_enabled:
        print(f"🔄 Dynamic VRAM scaling enabled: target {vram_scaler.target_min:.0%}-{vram_scaler.target_max:.0%}")
    
    # Prevent model cleanup during concurrent processing
    model_manager.set_concurrent_processing(True)
    
    try:
        # Process files with dynamic scaling
        results = []
        pending_tasks = tasks.copy()
        active_futures = {}
        
        # Start a background thread for time-based VRAM scaling
        scaling_stop_event = threading.Event()
        
        def time_based_scaling():
            nonlocal current_concurrent
            while not scaling_stop_event.is_set():
                if vram_scaler.should_check_scaling():
                    new_concurrent = vram_scaler.adjust_concurrency()
                    if new_concurrent != current_concurrent:
                        current_concurrent = new_concurrent
                        # Note: Can't add new futures here due to thread safety, will be handled in main loop
                time.sleep(5)  # Check every 5 seconds
        
        if vram_scaler.scaling_enabled:
            scaling_thread = threading.Thread(target=time_based_scaling, daemon=True)
            scaling_thread.start()
        
        # Use a smaller initial pool and expand dynamically
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:  # Large pool for dynamic scaling
            
            # Submit initial batch of tasks
            while len(active_futures) < current_concurrent and pending_tasks:
                task = pending_tasks.pop(0)
                future = executor.submit(process_single_file_sync, task, model_instance, whisper_language, language_mode, model_name)
                active_futures[future] = task
            
            # Process results and maintain optimal concurrency
            while active_futures or pending_tasks:
                if active_futures:
                    # Wait for at least one task to complete (no timeout to avoid TimeoutError)
                    try:
                        done_futures, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        
                        # Process all completed futures
                        for future in done_futures:
                            task = active_futures.pop(future)
                            
                            try:
                                result = future.result()
                                results.append(result)
                                print(f"✅ Completed {len(results)}/{len(tasks)}: {task.file.filename}")
                            except Exception as e:
                                print(f"❌ Exception for {task.file.filename}: {e}")
                                results.append({
                                    "filename": task.file.filename,
                                    "error": str(e),
                                    "status": "failed"
                                })
                            
                            # Check if we should adjust concurrency
                            new_concurrent = vram_scaler.file_completed()
                            if new_concurrent != current_concurrent:
                                current_concurrent = new_concurrent
                                print(f"🔄 Dynamic scaling: adjusting to {current_concurrent} concurrent files")
                            
                            # Submit new tasks to maintain desired concurrency
                            while len(active_futures) < current_concurrent and pending_tasks:
                                new_task = pending_tasks.pop(0)
                                new_future = executor.submit(process_single_file_sync, new_task, model_instance, whisper_language, language_mode, model_name)
                                active_futures[new_future] = new_task
                                
                    except Exception as e:
                        print(f"⚠️ Error in concurrent processing loop: {e}")
                        break
                else:
                    # No active futures but pending tasks - shouldn't happen
                    break
        
        # Stop the background scaling thread
        if vram_scaler.scaling_enabled:
            scaling_stop_event.set()
                        
    finally:
        # Always re-enable model cleanup after concurrent processing
        model_manager.set_concurrent_processing(False)
    
    # Clean up temporary files
    for task in tasks:
        if os.path.exists(task.file_path):
            os.unlink(task.file_path)
    
    # Collect metrics for return
    file_sizes = [task.file_size_mb for task in tasks]
    audio_lengths = [task.real_audio_length for task in tasks]
    audio_formats = [task.audio_format for task in tasks]
    
    successful_results = [r for r in results if r["status"] == "success"]
    processing_times = [r["processing_time"] for r in successful_results]
    transcript_lengths = [r["transcript_length"] for r in successful_results if "transcript_length" in r]
    language_confidences = [r["language_probability"] for r in successful_results]
    
    print(f"📊 Dynamic scaling summary: Started at {max_concurrent}, ended at {vram_scaler.current_concurrent} concurrent files")
    
    return results, {
        "file_sizes": file_sizes,
        "audio_lengths": audio_lengths,
        "audio_formats": audio_formats,
        "processing_times": processing_times,
        "transcript_lengths": transcript_lengths,
        "language_confidences": language_confidences
    }, vram_scaler.current_concurrent

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
            "output_directory": config.get_output_directory(),
            "concurrent_processing_enabled": config.is_concurrent_processing_enabled(),
            "max_concurrent_files": config.get_max_concurrent_files(),
            "adaptive_batching": config.is_adaptive_batching_enabled()
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
    
    # Check if concurrent processing is enabled
    use_concurrent = config.is_concurrent_processing_enabled() and len(files) > 1
    print(f"⚡ Processing mode: {'CONCURRENT' if use_concurrent else 'SEQUENTIAL'}")
    
    # Start metric collection
    model_load_start = time.time()
    
    # Load model in advance
    model_instance = model_manager.get_model(model_name)
    model_load_time = time.time() - model_load_start
    
    total_start_time = time.time()
    saved_files = []
    actual_concurrent_files = 1  # Default for sequential processing
    
    if use_concurrent:
        # Use new concurrent processing
        results, metrics_data, actual_concurrent_files = await process_files_concurrently(
            files, model_instance, whisper_language, language_mode, model_name
        )
        
        file_sizes = metrics_data["file_sizes"]
        processing_times = metrics_data["processing_times"]
        transcript_lengths = metrics_data["transcript_lengths"]
        language_confidences = metrics_data["language_confidences"]
        audio_formats = metrics_data["audio_formats"]
        real_audio_lengths = metrics_data["audio_lengths"]
        
        # Collect saved file paths
        saved_files = [r["saved_to"] for r in results if r["status"] == "success"]
        
    else:
        # Use original sequential processing for single files or when disabled
        results = []
        
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
                batch_size = model_instance['batch_size']

                print(f"🎯 Starting batched transcription for {file.filename}")
                print(f"🔢 Using batch_size: {batch_size}")
                print(f"🗣️ Language mode: {language_mode}")
                start_time = time.time()

                segments, info = model_instance['model'].transcribe(
                    file_path,
                    language=whisper_language,
                    task="transcribe",
                    batch_size=batch_size
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
        
        # CORRECTED: Use actual wall-clock time for concurrent processing metrics
        files_per_hour = (successful / total_time) * 3600 if total_time > 0 else 0
        files_per_day = files_per_hour * 24
        
        total_data_mb = sum(file_sizes)
        throughput_mb_per_s = total_data_mb / total_time if total_time > 0 else 0
        
        audio_minutes_per_hour = files_per_hour * avg_audio_length
        audio_minutes_per_day = audio_minutes_per_hour * 24
        
        # Real-time factor: use total audio vs total wall-clock time (not per-file average)
        total_audio_minutes = sum(real_audio_lengths)
        real_time_factor = (total_audio_minutes * 60 / total_time) if total_time > 0 else 0
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
            'real_time_factor': round(real_time_factor, 1),
            'concurrent_processing': use_concurrent,
            'max_concurrent_files': actual_concurrent_files
        }
        
        save_metrics_to_csv(metrics_data)
        
        print(f"\n📊 CORRECTED Performance Metrics (wall-clock time):")
        print(f"   GPU: {gpu_info['gpu_model']}")
        print(f"   Processing mode: {'CONCURRENT' if use_concurrent else 'SEQUENTIAL'}")
        if use_concurrent:
            print(f"   Max concurrent files: {actual_concurrent_files}")
        print(f"   Total batch time: {total_time:.2f}s")
        print(f"   Average audio length: {avg_audio_length:.2f} min")
        print(f"   Real-time factor: {real_time_factor:.1f}x (total audio vs wall-clock)")
        print(f"   Transcribable minutes/hour: {transcribable_minutes_per_hour:.1f}")
        print(f"   🚀 Files/hour: {files_per_hour:.1f} (CORRECTED for concurrency)")
    
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
