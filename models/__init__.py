"""
模型定义模块

论文实验只使用Qwen模型（通过Ollama部署）
"""
from .base_model import BaseContinualLearningModel

# Qwen模型（论文实验使用）
try:
    from .qwen_model import (
        Qwen2ForContinualLearning,
        OllamaQwen2ForContinualLearning,
        Qwen2OllamaWrapper
    )
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    Qwen2ForContinualLearning = None
    OllamaQwen2ForContinualLearning = None
    Qwen2OllamaWrapper = None

__all__ = [
    'BaseContinualLearningModel',
]

if QWEN_AVAILABLE:
    __all__.extend([
        'Qwen2ForContinualLearning',
        'OllamaQwen2ForContinualLearning',
        'Qwen2OllamaWrapper'
    ])

