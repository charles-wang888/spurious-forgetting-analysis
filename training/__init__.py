"""
训练模块
包含深层对齐训练和自适应缓解策略
"""
from .deep_alignment_trainer import DeepAlignmentTrainer
from .adaptive_mitigation import AdaptiveMitigationStrategy

__all__ = ['DeepAlignmentTrainer', 'AdaptiveMitigationStrategy']

