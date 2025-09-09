"""
Configuration management for SmallGPT
"""
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import yaml
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 10000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 0
    
    # Advanced options
    use_rope: bool = False  # Rotary position embeddings
    use_alibi: bool = False  # ALiBi positional bias
    use_rmsnorm: bool = False  # RMSNorm instead of LayerNorm
    use_swiglu: bool = False  # SwiGLU activation
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw'
    lr_scheduler: Optional[str] = 'linear_warmup'  # 'cosine', 'linear_warmup', None
    min_lr: float = 0.0
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Logging and evaluation
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Directories
    output_dir: str = './models'
    data_dir: str = './data'
    
    # Data
    max_seq_len: int = 1024
    num_workers: int = 4
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = 'smallgpt'
    experiment_name: str = 'default'
    
    # Mixed precision
    use_amp: bool = False
    
    def validate(self):
        """Validate training configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be between 0 and 1"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.warmup_steps <= self.max_steps, "warmup_steps must be <= max_steps"


@dataclass
class TokenizerConfig:
    """Tokenizer configuration"""
    vocab_size: int = 10000
    min_frequency: int = 2
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'bos_token': '<bos>',
        'eos_token': '<eos>',
    })
    
    # BPE parameters
    num_merges: int = 9000
    
    def validate(self):
        """Validate tokenizer configuration"""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.min_frequency > 0, "min_frequency must be positive"
        assert self.num_merges > 0, "num_merges must be positive"


@dataclass
class GenerationConfig:
    """Text generation configuration"""
    max_length: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Stopping criteria
    stop_tokens: List[str] = field(default_factory=list)
    max_time: Optional[float] = None
    
    def validate(self):
        """Validate generation configuration"""
        assert self.max_length > 0, "max_length must be positive"
        assert self.temperature > 0, "temperature must be positive"
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be positive"
        if self.top_p is not None:
            assert 0 < self.top_p <= 1, "top_p must be between 0 and 1"


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    def validate(self):
        """Validate all configurations"""
        self.model.validate()
        self.training.validate()
        self.tokenizer.validate()
        self.generation.validate()
        
        # Cross-validation
        assert self.model.vocab_size == self.tokenizer.vocab_size, \
            "Model and tokenizer vocab_size must match"
        assert self.model.max_seq_len == self.training.max_seq_len, \
            "Model and training max_seq_len must match"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    tokenizer_config = TokenizerConfig(**config_dict.get('tokenizer', {}))
    generation_config = GenerationConfig(**config_dict.get('generation', {}))
    
    config = Config(
        model=model_config,
        training=training_config,
        tokenizer=tokenizer_config,
        generation=generation_config
    )
    
    config.validate()
    return config


def save_config(config: Config, config_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        config_path: Path to save YAML configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'tokenizer': config.tokenizer.__dict__,
        'generation': config.generation.__dict__
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config()


def create_small_config() -> Config:
    """Create a small model configuration for testing."""
    config = Config()
    
    # Small model
    config.model.vocab_size = 5000
    config.model.embed_dim = 256
    config.model.num_heads = 4
    config.model.num_layers = 4
    config.model.max_seq_len = 512
    
    # Fast training
    config.training.batch_size = 16
    config.training.max_steps = 10000
    config.training.warmup_steps = 500
    config.training.max_seq_len = 512
    
    # Small tokenizer
    config.tokenizer.vocab_size = 5000
    config.tokenizer.num_merges = 4500
    
    config.validate()
    return config


def create_medium_config() -> Config:
    """Create a medium model configuration."""
    config = Config()
    
    # Medium model
    config.model.vocab_size = 15000
    config.model.embed_dim = 768
    config.model.num_heads = 12
    config.model.num_layers = 8
    config.model.max_seq_len = 1024
    
    # Training
    config.training.batch_size = 24
    config.training.max_steps = 50000
    config.training.warmup_steps = 2000
    config.training.max_seq_len = 1024
    
    # Tokenizer
    config.tokenizer.vocab_size = 15000
    config.tokenizer.num_merges = 14000
    
    config.validate()
    return config


def create_large_config() -> Config:
    """Create a large model configuration."""
    config = Config()
    
    # Large model
    config.model.vocab_size = 30000
    config.model.embed_dim = 1024
    config.model.num_heads = 16
    config.model.num_layers = 12
    config.model.max_seq_len = 2048
    
    # Training
    config.training.batch_size = 16
    config.training.max_steps = 100000
    config.training.warmup_steps = 4000
    config.training.max_seq_len = 2048
    
    # Tokenizer
    config.tokenizer.vocab_size = 30000
    config.tokenizer.num_merges = 29000
    
    config.validate()
    return config
