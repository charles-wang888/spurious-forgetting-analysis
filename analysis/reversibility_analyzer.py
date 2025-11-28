"""
可逆性分析工具
判断遗忘是否可逆，量化可逆性
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ReversibilityAnalyzer:
    """可逆性分析工具"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reversibility_history = {}  # {task_id: [scores]}
    
    def analyze_reversibility(
        self,
        representation_before: torch.Tensor,
        representation_after: torch.Tensor,
        performance_before: float,
        performance_after: float
    ) -> Tuple[bool, float]:
        """
        分析可逆性
        
        Returns:
            (is_reversible, reversibility_score)
        """
        # 计算表示相似度
        rep_before_mean = representation_before.mean(dim=0)
        rep_after_mean = representation_after.mean(dim=0)
        
        rep_before_norm = rep_before_mean / (torch.norm(rep_before_mean) + 1e-8)
        rep_after_norm = rep_after_mean / (torch.norm(rep_after_mean) + 1e-8)
        
        similarity = torch.dot(rep_before_norm, rep_after_norm).item()
        similarity = (similarity + 1) / 2  # 映射到[0, 1]
        
        # 计算性能下降
        performance_drop = max(0, performance_before - performance_after) / (performance_before + 1e-8)
        
        # 可逆性分数：相似度高 + 性能下降大 = 高可逆性
        if performance_drop < 1e-6:
            reversibility = 1.0
        else:
            reversibility = similarity / (1 + performance_drop)
        
        is_reversible = reversibility >= 0.6
        
        return is_reversible, reversibility
    
    def track_reversibility(
        self,
        task_id: int,
        reversibility_score: float
    ):
        """跟踪可逆性分数"""
        if task_id not in self.reversibility_history:
            self.reversibility_history[task_id] = []
        self.reversibility_history[task_id].append(reversibility_score)
    
    def plot_reversibility_map(
        self,
        save_path: str = None
    ):
        """绘制可逆性地图"""
        if not self.reversibility_history:
            return
        
        num_tasks = len(self.reversibility_history)
        max_length = max(len(scores) for scores in self.reversibility_history.values())
        
        reversibility_matrix = np.zeros((num_tasks, max_length))
        
        for task_id, scores in self.reversibility_history.items():
            reversibility_matrix[task_id, :len(scores)] = scores
        
        plt.figure(figsize=(12, 8))
        plt.imshow(reversibility_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Reversibility Score')
        plt.xlabel('Time Step')
        plt.ylabel('Task ID')
        plt.title('Reversibility Map')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()

