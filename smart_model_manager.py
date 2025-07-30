import torch
import time
import threading
from faster_whisper import WhisperModel, BatchedInferencePipeline  # ← BatchedInferencePipeline hinzugefügt
import pynvml
import os
from pathlib import Path
from config_manager import WhisperConfigManager

class SmartModelManager:
    def __init__(self, config_path="whisper_config.yaml"):
        # Load configuration
        self.config = WhisperConfigManager(config_path)
        
        # Validate config on startup
        if not self.config.validate_config():
            raise RuntimeError("Invalid configuration - cannot start model manager")
        
        self.models = {}
        self.last_used = {}
        self.lock = threading.Lock()
        self.concurrent_processing_active = False  # Flag to prevent cleanup during concurrent processing
        
        # Use timeout from config
        self.timeout_minutes = self.config.get_model_timeout_minutes()
        
        # Set environment for CUDA 12 libraries
        lib_path = Path.home() / "whisper-batch" / "lib" / "python3.10" / "site-packages"
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}/nvidia/cublas/lib:{lib_path}/nvidia/cudnn/lib:{lib_path}/ctranslate2.libs:/usr/local/cuda-11.8/lib64:"
        
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print(f"🔧 SmartModelManager initialized:")
        print(f"   Default model: {self.config.get_default_model()}")
        print(f"   Model timeout: {self.timeout_minutes} minutes")
        print(f"   Batch size: {self.config.get_model_batch_size()}")
        
        # Start auto-cleanup thread
        self._start_cleanup_thread()
    
    def get_model(self, model_name=None):
        """Get model - uses config default if no model specified"""
        if model_name is None:
            model_name = self.config.get_default_model()
        
        # Validate requested model
        if not self.config.is_model_supported(model_name):
            raise ValueError(f"Unsupported model: {model_name}. Supported: {self.get_supported_models()}")
        
        with self.lock:
            if model_name in self.models:
                self.last_used[model_name] = time.time()
                print(f"♻️ Using already loaded {model_name} model")
                return self.models[model_name]
            
            return self._load_model(model_name)
    
    def _load_model(self, model_name):
        print(f"📥 Loading {model_name} model with batching...")
        
        # Standard Model laden
        base_model = WhisperModel(
            model_name, 
            device="cuda", 
            compute_type="float16"
        )
        
        # Batch size aus Config holen
        batch_size = self.config.get_model_batch_size()
        
        # BatchedInferencePipeline erstellen (OHNE batch_size Parameter!)
        batched_model = BatchedInferencePipeline(model=base_model)
        
        model_instance = {
            'model': batched_model,      # ← Jetzt BatchedInferencePipeline statt WhisperModel
            'base_model': base_model,    # Backup falls benötigt
            'batch_size': batch_size,
            'model_name': model_name,
            'loaded_at': time.time(),
            'vram_usage_gb': self.config.model.vram_usage_gb
        }
        
        self.models[model_name] = model_instance
        self.last_used[model_name] = time.time()
        
        # Log VRAM status
        info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        vram_used_gb = info.used / (1024**3)
        
        print(f"✅ {model_name} model loaded with BatchedInferencePipeline")
        print(f"   Batch size: {batch_size}")
        print(f"   VRAM used: {vram_used_gb:.1f}GB")
        print(f"   Expected VRAM: {self.config.model.vram_usage_gb}GB")
        
        return model_instance
    
    def get_default_model_name(self):
        """Get the default model name from config"""
        return self.config.get_default_model()
    
    def get_supported_models(self):
        """Get list of supported models"""
        return [
            # Large models (recommended)
            "large-v2", "large-v3", "large-v3-turbo", "large-v3-german",
            # Medium models
            "medium", "medium.en",
            # Small models  
            "small", "small.en",
            # Base models
            "base", "base.en",
            # Tiny models
            "tiny", "tiny.en"
        ]
    
    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(30)  # Check every 30 seconds
                self._cleanup_inactive_models()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_inactive_models(self):
        current_time = time.time()
        timeout_seconds = self.timeout_minutes * 60
        
        with self.lock:
            # Skip cleanup if concurrent processing is active
            if self.concurrent_processing_active:
                return
                
            models_to_remove = []
            
            for model_name, last_used_time in self.last_used.items():
                if current_time - last_used_time > timeout_seconds:
                    models_to_remove.append(model_name)
            
            for model_name in models_to_remove:
                print(f"🗑️ Unloading {model_name} after {self.timeout_minutes}min inactivity")
                del self.models[model_name]
                del self.last_used[model_name]
                torch.cuda.empty_cache()
                
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_used_gb = info.used / (1024**3)
                print(f"🧹 VRAM after cleanup: {vram_used_gb:.1f}GB")
    
    def set_concurrent_processing(self, active: bool):
        """Set concurrent processing flag to prevent/allow model cleanup"""
        with self.lock:
            self.concurrent_processing_active = active
            if active:
                print(f"🔒 Model cleanup disabled during concurrent processing")
            else:
                print(f"🔓 Model cleanup re-enabled after concurrent processing")
    
    def force_unload_all(self):
        """Unload all models for other containers"""
        with self.lock:
            if self.models:
                print("🚨 Force unloading all models for other workloads")
                model_names = list(self.models.keys())
                self.models.clear()
                self.last_used.clear()
                torch.cuda.empty_cache()
                
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_used_gb = info.used / (1024**3)
                print(f"✅ All models unloaded. VRAM: {vram_used_gb:.1f}GB")
                print(f"   Unloaded models: {', '.join(model_names)}")
            else:
                print("ℹ️ No models currently loaded")
    
    def get_model_status(self):
        """Get current model status"""
        with self.lock:
            status = {
                'loaded_models': list(self.models.keys()),
                'default_model': self.config.get_default_model(),
                'supported_models': self.get_supported_models(),
                'model_timeout_minutes': self.timeout_minutes,
                'config_batch_size': self.config.get_model_batch_size()
            }
            
            # Add last used times
            for model_name in self.models.keys():
                last_used = self.last_used.get(model_name, 0)
                inactive_time = (time.time() - last_used) / 60  # minutes
                status[f'{model_name}_inactive_minutes'] = round(inactive_time, 1)
            
            return status
    
    def reload_config(self):
        """Reload configuration (useful for runtime config changes)"""
        old_default = self.config.get_default_model()
        old_timeout = self.timeout_minutes
        
        self.config.reload_config()
        self.timeout_minutes = self.config.get_model_timeout_minutes()
        
        new_default = self.config.get_default_model()
        
        print(f"🔄 Config reloaded:")
        print(f"   Default model: {old_default} → {new_default}")
        print(f"   Timeout: {old_timeout} → {self.timeout_minutes} minutes")
        
        # Validate new config
        if not self.config.validate_config():
            print("⚠️ Warning: New config validation failed")