import yaml
from dataclasses import dataclass
from typing import Dict
from pathlib import Path

@dataclass
class ModelConfig:
    default_model: str
    vram_usage_gb: float
    batch_size: int
    expected_throughput: int

@dataclass
class SystemConfig:
    gpu_device: str
    max_vram_usage: float
    shared_gpu: bool

@dataclass
class SingleWorkerConfig:
    lazy_loading: bool
    model_timeout_minutes: int

@dataclass
class LanguageConfig:
    default_language: str
    auto_detect: bool

@dataclass
class OutputConfig:
    directory: str

@dataclass
class APIConfig:
    host: str
    port: int
    max_batch_size: int

@dataclass
class ConcurrentProcessingConfig:
    enabled: bool
    max_concurrent_files: int
    adaptive_batching: bool
    vram_threshold: float
    auto_gpu_scaling: bool
    base_model_vram_gb: float
    overhead_per_file_gb: float
    safety_buffer_gb: float
    dynamic_vram_scaling: bool
    target_vram_min: float
    target_vram_max: float
    emergency_vram_max: float
    scaling_step_size: int
    scaling_check_interval: int
    scaling_time_interval: int

class WhisperConfigManager:
    def __init__(self, config_path="whisper_config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config = self._load_config()
        
        # Create typed config objects
        self.system = SystemConfig(**self.config['system'])
        self.single_worker = SingleWorkerConfig(**self.config['single_worker'])
        self.model = ModelConfig(**self.config['model'])
        self.language = LanguageConfig(**self.config['language'])
        self.output = OutputConfig(**self.config['output'])
        self.api = APIConfig(**self.config['api'])
        
        # Handle concurrent processing config (with defaults for backward compatibility)
        concurrent_config = self.config.get('concurrent_processing', {
            'enabled': False,
            'max_concurrent_files': 4,
            'adaptive_batching': True,
            'vram_threshold': 0.8,
            'auto_gpu_scaling': True,
            'base_model_vram_gb': 4.7,
            'overhead_per_file_gb': 0.4,
            'safety_buffer_gb': 1.0,
            'dynamic_vram_scaling': False,
            'target_vram_min': 0.75,
            'target_vram_max': 0.85,
            'emergency_vram_max': 0.90,
            'scaling_step_size': 1,
            'scaling_check_interval': 1,
            'scaling_time_interval': 30
        })
        self.concurrent_processing = ConcurrentProcessingConfig(**concurrent_config)
        
    def _load_config(self) -> Dict:
        """Load YAML config file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def get_default_model(self) -> str:
        """Get the configured default model"""
        return self.model.default_model
    
    def get_default_language(self) -> str:
        """Get the configured default language"""
        return self.language.default_language
    
    def is_auto_detect_enabled(self) -> bool:
        """Check if automatic language detection is enabled"""
        return self.language.auto_detect
    
    def get_model_batch_size(self) -> int:
        """Get the configured batch size"""
        return self.model.batch_size
    
    def get_model_timeout_minutes(self) -> int:
        """Get model timeout in minutes"""
        return self.single_worker.model_timeout_minutes
    
    def get_max_batch_size(self) -> int:
        """Get maximum allowed batch size"""
        return self.api.max_batch_size
    
    def get_api_host_port(self) -> tuple:
        """Get API host and port"""
        return (self.api.host, self.api.port)
    
    def get_output_directory(self) -> str:
        """Get the configured output directory"""
        return self.output.directory
    
    def is_concurrent_processing_enabled(self) -> bool:
        """Check if concurrent processing is enabled"""
        return self.concurrent_processing.enabled
    
    def get_max_concurrent_files(self) -> int:
        """Get maximum concurrent files setting"""
        return self.concurrent_processing.max_concurrent_files
    
    def is_adaptive_batching_enabled(self) -> bool:
        """Check if adaptive batching is enabled"""
        return self.concurrent_processing.adaptive_batching
    
    def get_vram_threshold(self) -> float:
        """Get VRAM threshold for concurrent processing"""
        return self.concurrent_processing.vram_threshold
    
    def is_auto_gpu_scaling_enabled(self) -> bool:
        """Check if automatic GPU scaling is enabled"""
        return self.concurrent_processing.auto_gpu_scaling
    
    def is_model_supported(self, model_name: str) -> bool:
        """Check if model is supported"""
        supported_models = [
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
        return model_name in supported_models
    
    def validate_config(self) -> bool:
        """Validate the configuration"""
        try:
            # Validate model
            if not self.is_model_supported(self.model.default_model):
                raise ValueError(f"Unsupported model: {self.model.default_model}")
            
            # Validate batch size
            if self.model.batch_size < 1 or self.model.batch_size > 2048:
                raise ValueError(f"Invalid batch size: {self.model.batch_size}")
            
            # Validate VRAM with model-specific recommendations
            model_vram_recommendations = {
                "large-v3": 4.7, "large-v3-turbo": 4.5, "large-v3-german": 4.8, "large-v2": 4.7,
                "medium": 2.4, "medium.en": 2.4,
                "small": 1.2, "small.en": 1.2,
                "base": 0.8, "base.en": 0.8,
                "tiny": 0.4, "tiny.en": 0.4
            }
            
            expected_vram = model_vram_recommendations.get(self.model.default_model, 4.7)
            if abs(self.model.vram_usage_gb - expected_vram) > 1.0:
                print(f"⚠️ VRAM setting ({self.model.vram_usage_gb}GB) differs from recommendation for {self.model.default_model}: {expected_vram}GB")
            
            if self.model.vram_usage_gb < 0.2 or self.model.vram_usage_gb > 24:
                raise ValueError(f"Invalid VRAM usage: {self.model.vram_usage_gb}GB")
            
            # Validate timeout
            if self.single_worker.model_timeout_minutes < 1:
                raise ValueError(f"Invalid timeout: {self.single_worker.model_timeout_minutes} minutes")
            
            # Validate and create output directory
            output_path = Path(self.output.directory)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Output directory: {output_path.absolute()}")
            except Exception as e:
                raise ValueError(f"Cannot create output directory {self.output.directory}: {e}")
            
            print(f"✅ Config validation successful")
            print(f"   Default model: {self.model.default_model}")
            print(f"   Expected VRAM: {expected_vram}GB, Configured: {self.model.vram_usage_gb}GB")
            print(f"   Default language: {self.language.default_language}")
            print(f"   Auto-detect enabled: {self.language.auto_detect}")
            print(f"   Batch size: {self.model.batch_size}")
            print(f"   Model timeout: {self.single_worker.model_timeout_minutes} minutes")
            print(f"   Output directory: {output_path.absolute()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Config validation failed: {e}")
            return False
    
    def reload_config(self):
        """Reload configuration from file"""
        self.__init__(self.config_path)
