"""
动态跟踪工具
跟踪训练过程中的表示空间变化
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class RepresentationTracker:
    """表示空间动态跟踪工具"""
    
    def __init__(self, track_frequency: int = 10):
        self.track_frequency = track_frequency
        self.representation_snapshots = defaultdict(dict)  # {task_id: {epoch: representation}}
        self.performance_snapshots = defaultdict(dict)  # {task_id: {epoch: performance}}
    
    def should_track(self, epoch: int) -> bool:
        """判断是否应该记录快照"""
        return epoch % self.track_frequency == 0
    
    def record_snapshot(
        self,
        task_id: int,
        epoch: int,
        representation: torch.Tensor,
        performance: float
    ):
        """记录快照"""
        if self.should_track(epoch) or epoch == 0:
            # 存储平均表示以节省内存
            rep_mean = representation.mean(dim=0).cpu()
            self.representation_snapshots[task_id][epoch] = rep_mean
            self.performance_snapshots[task_id][epoch] = performance
    
    def get_representation_trajectory(
        self,
        task_id: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """获取表示轨迹"""
        if task_id not in self.representation_snapshots:
            return [], []
        
        epochs = sorted(self.representation_snapshots[task_id].keys())
        representations = [self.representation_snapshots[task_id][e] for e in epochs]
        
        return representations, epochs
    
    def compute_representation_drift(
        self,
        task_id: int
    ) -> float:
        """计算表示漂移（从初始到最终的变化）"""
        if task_id not in self.representation_snapshots:
            return 0.0
        
        epochs = sorted(self.representation_snapshots[task_id].keys())
        if len(epochs) < 2:
            return 0.0
        
        initial_rep = self.representation_snapshots[task_id][epochs[0]]
        final_rep = self.representation_snapshots[task_id][epochs[-1]]
        
        # 计算余弦距离
        initial_norm = initial_rep / (torch.norm(initial_rep) + 1e-8)
        final_norm = final_rep / (torch.norm(final_rep) + 1e-8)
        
        cosine_sim = torch.dot(initial_norm, final_norm).item()
        drift = 1 - cosine_sim  # 转换为距离
        
        return drift
    
    def get_key_timepoints(
        self,
        task_id: int,
        threshold: float = 0.1
    ) -> List[int]:
        """获取关键时间点（表示发生显著变化的时间点）"""
        if task_id not in self.representation_snapshots:
            return []
        
        epochs = sorted(self.representation_snapshots[task_id].keys())
        if len(epochs) < 2:
            return []
        
        key_timepoints = [epochs[0]]
        
        for i in range(1, len(epochs)):
            prev_rep = self.representation_snapshots[task_id][epochs[i-1]]
            curr_rep = self.representation_snapshots[task_id][epochs[i]]
            
            # 计算变化
            prev_norm = prev_rep / (torch.norm(prev_rep) + 1e-8)
            curr_norm = curr_rep / (torch.norm(curr_rep) + 1e-8)
            
            change = 1 - torch.dot(prev_norm, curr_norm).item()
            
            if change > threshold:
                key_timepoints.append(epochs[i])
        
        return key_timepoints
    
    def clear_snapshots(self, task_id: Optional[int] = None):
        """清除快照（释放内存）"""
        if task_id is None:
            self.representation_snapshots.clear()
            self.performance_snapshots.clear()
        else:
            if task_id in self.representation_snapshots:
                del self.representation_snapshots[task_id]
            if task_id in self.performance_snapshots:
                del self.performance_snapshots[task_id]

