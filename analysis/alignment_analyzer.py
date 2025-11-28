"""
对齐分析工具
分析不同层之间的对齐关系，量化对齐度
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class AlignmentAnalyzer:
    """对齐分析工具"""
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.alignment_history = {}  # {task_id: {layer_name: [scores]}}
    
    def analyze_layer_alignment(
        self,
        layer_representations: Dict[str, torch.Tensor],
        output_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        分析各层的对齐度
        
        Args:
            layer_representations: {layer_name: representation_tensor}
            output_weights: 输出层权重
        
        Returns:
            {layer_name: alignment_score}
        """
        alignment_scores = {}
        
        for layer_name, representations in layer_representations.items():
            # 计算对齐度（简化版）
            rep_mean = representations.mean(dim=0)
            weights_mean = output_weights.mean(dim=0)
            
            # 余弦相似度
            rep_norm = rep_mean / (torch.norm(rep_mean) + 1e-8)
            weights_norm = weights_mean / (torch.norm(weights_mean) + 1e-8)
            
            alignment = torch.dot(rep_norm, weights_norm).item()
            alignment = (alignment + 1) / 2  # 映射到[0, 1]
            
            alignment_scores[layer_name] = alignment
        
        return alignment_scores
    
    def track_alignment_over_time(
        self,
        task_id: int,
        alignment_scores: Dict[str, float]
    ):
        """跟踪对齐度随时间的变化"""
        if task_id not in self.alignment_history:
            self.alignment_history[task_id] = {}
        
        for layer_name, score in alignment_scores.items():
            if layer_name not in self.alignment_history[task_id]:
                self.alignment_history[task_id][layer_name] = []
            self.alignment_history[task_id][layer_name].append(score)
    
    def plot_alignment_heatmap(
        self,
        task_id: int,
        save_path: str = None
    ):
        """绘制对齐度热力图"""
        if task_id not in self.alignment_history:
            return
        
        # 准备数据
        layer_names = list(self.alignment_history[task_id].keys())
        num_layers = len(layer_names)
        num_timepoints = max(len(scores) for scores in self.alignment_history[task_id].values())
        
        heatmap_data = np.zeros((num_layers, num_timepoints))
        
        for i, layer_name in enumerate(layer_names):
            scores = self.alignment_history[task_id][layer_name]
            heatmap_data[i, :len(scores)] = scores
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            xticklabels=range(num_timepoints),
            yticklabels=layer_names,
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            annot=False
        )
        plt.title(f'Alignment Heatmap for Task {task_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Layer')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def detect_alignment_breakdown(
        self,
        task_id: int,
        threshold: float = 0.3
    ) -> List[str]:
        """
        检测对齐破坏的层
        
        Args:
            task_id: 任务ID
            threshold: 对齐度下降阈值
        
        Returns:
            对齐破坏的层名称列表
        """
        if task_id not in self.alignment_history:
            return []
        
        broken_layers = []
        
        for layer_name, scores in self.alignment_history[task_id].items():
            if len(scores) >= 2:
                initial_score = scores[0]
                final_score = scores[-1]
                drop = initial_score - final_score
                
                if drop > threshold:
                    broken_layers.append(layer_name)
        
        return broken_layers

