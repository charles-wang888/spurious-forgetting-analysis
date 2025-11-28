"""
辅助函数
"""
import torch
import numpy as np
from typing import List, Dict

def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """获取可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

