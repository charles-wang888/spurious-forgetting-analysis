"""
可逆性指标计算模块
用于判断遗忘是否可逆
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

class ReversibilityScore:
    """
    可逆性指标计算器
    测量遗忘的可逆程度
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def compute_representation_similarity(
        self,
        rep1: torch.Tensor,
        rep2: torch.Tensor,
        method: str = "cka"
    ) -> float:
        """
        计算两个表示之间的相似度
        
        Args:
            rep1: 表示1 [batch_size, hidden_dim]
            rep2: 表示2 [batch_size, hidden_dim]
            method: 计算方法 ('cka', 'cosine', 'euclidean')
        
        Returns:
            相似度分数 [0, 1]
        """
        if method == "cosine":
            # 平均后计算余弦相似度
            rep1_mean = rep1.mean(dim=0)
            rep2_mean = rep2.mean(dim=0)
            
            rep1_norm = rep1_mean / (torch.norm(rep1_mean) + 1e-8)
            rep2_norm = rep2_mean / (torch.norm(rep2_mean) + 1e-8)
            
            similarity = torch.dot(rep1_norm, rep2_norm).item()
            return (similarity + 1) / 2  # 映射到[0, 1]
        
        elif method == "cka":
            # Centered Kernel Alignment
            # 可以使用完整实现（analysis.cka_implementation）或简化版本
            try:
                # 尝试使用完整实现
                from analysis.cka_implementation import linear_cka
                return linear_cka(rep1, rep2, debiased=False)
            except ImportError:
                # 回退到简化版本
                rep1_centered = rep1 - rep1.mean(dim=0, keepdim=True)
                rep2_centered = rep2 - rep2.mean(dim=0, keepdim=True)
                
                # 计算Gram矩阵
                K1 = torch.mm(rep1_centered, rep1_centered.t())
                K2 = torch.mm(rep2_centered, rep2_centered.t())
                
                # CKA分数
                hsic = torch.trace(torch.mm(K1, K2))
                norm1 = torch.trace(torch.mm(K1, K1))
                norm2 = torch.trace(torch.mm(K2, K2))
                
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cka = hsic / (torch.sqrt(norm1 * norm2) + 1e-8)
                    return cka.item()
                else:
                    return 0.0
        
        elif method == "euclidean":
            # 欧氏距离（转换为相似度）
            rep1_mean = rep1.mean(dim=0)
            rep2_mean = rep2.mean(dim=0)
            
            distance = torch.norm(rep1_mean - rep2_mean).item()
            # 使用指数函数转换为相似度
            similarity = np.exp(-distance / 100.0)  # 可调整缩放因子
            return similarity
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_reversibility_score(
        self,
        representation_before: torch.Tensor,
        representation_after: torch.Tensor,
        performance_before: float,
        performance_after: float,
        method: str = "combined"
    ) -> float:
        """
        计算可逆性分数
        
        可逆性 = f(表示空间变化, 性能变化)
        如果表示空间变化小但性能下降大，说明可能是虚假遗忘（可逆）
        如果表示空间变化大且性能下降大，说明可能是真实遗忘（不可逆）
        
        Args:
            representation_before: 训练前的表示
            representation_after: 训练后的表示
            performance_before: 训练前的性能
            performance_after: 训练后的性能
            method: 计算方法
        
        Returns:
            可逆性分数 [0, 1]，越高表示越可逆
        """
        # 计算表示空间相似度
        rep_similarity = self.compute_representation_similarity(
            representation_before, representation_after, method="cosine"
        )
        
        # 计算性能下降
        performance_drop = max(0, performance_before - performance_after) / (performance_before + 1e-8)
        
        if method == "combined":
            # 组合方法：表示相似度高 + 性能下降大 = 高可逆性
            # 表示相似度低 + 性能下降大 = 低可逆性
            if performance_drop < 1e-6:
                # 没有性能下降，可逆性为1
                return 1.0
            
            # 可逆性 = 表示相似度 / (1 + 性能下降)
            reversibility = rep_similarity / (1 + performance_drop)
            return float(reversibility)
        
        elif method == "ratio":
            # 比率方法：可逆性 = 表示相似度 / 性能下降
            if performance_drop < 1e-6:
                return 1.0
            
            ratio = rep_similarity / (performance_drop + 1e-8)
            # 归一化到[0, 1]
            reversibility = min(ratio, 1.0)
            return float(reversibility)
        
        elif method == "difference":
            # 差值方法：可逆性 = 表示相似度 - 性能下降
            reversibility = rep_similarity - performance_drop
            return max(0.0, min(1.0, reversibility))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_reversibility(
        self,
        representation_before: torch.Tensor,
        representation_after: torch.Tensor,
        performance_before: float,
        performance_after: float,
        threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        预测遗忘是否可逆
        
        Args:
            representation_before: 训练前的表示
            representation_after: 训练后的表示
            performance_before: 训练前的性能
            performance_after: 训练后的性能
            threshold: 可逆性阈值
        
        Returns:
            (is_reversible, reversibility_score)
        """
        score = self.compute_reversibility_score(
            representation_before,
            representation_after,
            performance_before,
            performance_after
        )
        is_reversible = score >= threshold
        return is_reversible, score

