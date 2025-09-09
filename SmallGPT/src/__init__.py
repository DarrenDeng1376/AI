"""
SmallGPT - A minimal GPT implementation for educational purposes
"""

__version__ = "0.1.0"
__author__ = "SmallGPT Team"

from .model.gpt import SmallGPT
from .utils.tokenizer import SimpleTokenizer
from .utils.config import Config, load_config, create_default_config
from .utils.sampling import generate_text

__all__ = [
    'SmallGPT',
    'SimpleTokenizer', 
    'Config',
    'load_config',
    'create_default_config',
    'generate_text'
]
