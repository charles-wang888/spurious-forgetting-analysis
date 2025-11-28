"""
评测指标模块
"""
from .alignment_score import AlignmentScore
from .reversibility import ReversibilityScore
from .forgetting_metrics import (
    compute_forgetting_measure,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_average_accuracy
)
from .evaluation import Evaluator

__all__ = [
    'AlignmentScore',
    'ReversibilityScore',
    'compute_forgetting_measure',
    'compute_backward_transfer',
    'compute_forward_transfer',
    'compute_average_accuracy',
    'Evaluator'
]

