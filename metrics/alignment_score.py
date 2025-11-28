"""
对齐度指标计算模块
用于量化任务对齐的程度
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

class AlignmentScore:
    """
    对齐度指标计算器
    测量输出层与内部表示的对齐程度
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.hooks = []
        self.activations = {}
    
    def register_hooks(self, layer_names: List[str]):
        """注册钩子函数以捕获中间层表示"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(get_activation(name)))
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_alignment_score(
        self,
        representations: torch.Tensor,
        output_weights: torch.Tensor,
        method: str = "cosine"
    ) -> float:
        """
        计算对齐度分数
        
        Args:
            representations: 内部表示 [batch_size, hidden_dim]
            output_weights: 输出层权重 [num_classes, hidden_dim]
            method: 计算方法 ('cosine', 'projection', 'correlation')
        
        Returns:
            对齐度分数 [0, 1]
        """
        if method == "cosine":
            # 使用余弦相似度
            # 计算表示与输出权重的平均相似度
            rep_mean = representations.mean(dim=0)  # [hidden_dim]
            weights_mean = output_weights.mean(dim=0)  # [hidden_dim]
            
            # 归一化
            rep_norm = rep_mean / (torch.norm(rep_mean) + 1e-8)
            weights_norm = weights_mean / (torch.norm(weights_mean) + 1e-8)
            
            alignment = torch.dot(rep_norm, weights_norm).item()
            # 映射到[0, 1]
            alignment = (alignment + 1) / 2
            return alignment
        
        elif method == "projection":
            # 使用投影方法
            # 计算表示在输出权重空间上的投影
            rep_mean = representations.mean(dim=0).unsqueeze(0)  # [1, hidden_dim]
            weights = output_weights  # [num_classes, hidden_dim]
            
            # 投影
            projections = torch.mm(rep_mean, weights.t())  # [1, num_classes]
            projection_norm = torch.norm(projections)
            rep_norm = torch.norm(rep_mean)
            
            if rep_norm > 1e-8:
                alignment = (projection_norm / rep_norm).item()
            else:
                alignment = 0.0
            
            return min(alignment, 1.0)
        
        elif method == "correlation":
            # 使用相关性方法
            rep_mean = representations.mean(dim=0).cpu().numpy()
            weights_mean = output_weights.mean(dim=0).cpu().numpy()
            
            # 计算皮尔逊相关系数
            rep_centered = rep_mean - np.mean(rep_mean)
            weights_centered = weights_mean - np.mean(weights_mean)
            
            numerator = np.dot(rep_centered, weights_centered)
            denominator = (np.linalg.norm(rep_centered) * np.linalg.norm(weights_centered) + 1e-8)
            
            correlation = numerator / denominator
            # 映射到[0, 1]
            alignment = (correlation + 1) / 2
            return float(alignment)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_layer_alignment(
        self,
        layer_representations: Dict[str, torch.Tensor],
        output_layer_name: str = "classifier"
    ) -> Dict[str, float]:
        """
        计算各层与输出层的对齐度
        
        Args:
            layer_representations: 各层的表示 {layer_name: tensor}
            output_layer_name: 输出层名称
        
        Returns:
            各层的对齐度分数 {layer_name: score}
        """
        alignment_scores = {}
        
        # 获取输出层权重
        output_layer = None
        for name, module in self.model.named_modules():
            if output_layer_name in name.lower() and isinstance(module, nn.Linear):
                output_layer = module
                break
        
        if output_layer is None:
            # 如果没有找到输出层，使用最后一层
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    output_layer = module
                    break
        
        if output_layer is None:
            return alignment_scores
        
        output_weights = output_layer.weight.data  # [num_classes, hidden_dim]
        
        # 计算每层的对齐度
        for layer_name, representations in layer_representations.items():
            if representations.dim() > 2:
                # 如果是多维张量，需要flatten或pooling
                if representations.dim() == 3:  # [batch, seq_len, hidden]
                    representations = representations.mean(dim=1)  # 平均池化
                else:
                    representations = representations.view(representations.size(0), -1)
            
            # 确保维度匹配
            if representations.size(1) != output_weights.size(1):
                # 如果维度不匹配，使用线性投影
                if representations.size(1) > output_weights.size(1):
                    # 降维
                    proj = nn.Linear(representations.size(1), output_weights.size(1)).to(self.device)
                    representations = proj(representations)
                else:
                    # 升维（填充零）
                    padding = torch.zeros(
                        representations.size(0),
                        output_weights.size(1) - representations.size(1)
                    ).to(self.device)
                    representations = torch.cat([representations, padding], dim=1)
            
            score = self.compute_alignment_score(representations, output_weights)
            alignment_scores[layer_name] = score
        
        return alignment_scores
    
    def compute_alignment_change(
        self,
        alignment_before: Dict[str, float],
        alignment_after: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算对齐度的变化
        
        Args:
            alignment_before: 训练前的对齐度
            alignment_after: 训练后的对齐度
        
        Returns:
            对齐度变化 {layer_name: change}
        """
        changes = {}
        for layer_name in alignment_before.keys():
            if layer_name in alignment_after:
                changes[layer_name] = alignment_after[layer_name] - alignment_before[layer_name]
        return changes
    
    def compute_alignment_depth(
        self,
        token_alignment_scores: torch.Tensor,
        tau_deep: float = 0.7
    ) -> int:
        """
        计算对齐深度
        D(θ, T) = max{k : A_t(θ, T) ≥ τ_deep for all t ≤ k}
        
        Args:
            token_alignment_scores: token级别的对齐分数 [batch_size, seq_len] 或 [seq_len]
            tau_deep: 深层对齐阈值
            
        Returns:
            对齐深度D
        """
        if token_alignment_scores.dim() == 1:
            # [seq_len]
            scores = token_alignment_scores
        else:
            # [batch_size, seq_len] -> 平均
            scores = torch.mean(token_alignment_scores, dim=0)
        
        # 找到满足阈值的最长连续序列
        depth = 0
        for t in range(scores.size(0)):
            if scores[t].item() >= tau_deep:
                depth = t + 1
            else:
                break
        
        return depth
    
    def compute_token_level_alignment(
        self,
        hidden_states: torch.Tensor,
        output_weights: torch.Tensor,
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个token位置的对齐分数
        A_t(θ, T) = cosine(H_L(x, t) W_out, y_true, t)
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            output_weights: 输出层权重 [num_classes, hidden_dim]
            true_labels: 真实标签 [batch_size, seq_len] 或 [batch_size]
            
        Returns:
            对齐分数 [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 计算每个token位置的输出
        outputs = torch.matmul(hidden_states, output_weights.t())  # [batch_size, seq_len, num_classes]
        
        # 处理标签
        if true_labels.dim() == 1:
            true_labels = true_labels.unsqueeze(1).expand(-1, seq_len)
        
        # 计算每个位置的余弦相似度（简化处理）
        alignment_scores = []
        for t in range(seq_len):
            output_t = outputs[:, t, :]  # [batch_size, num_classes]
            label_t = true_labels[:, t]  # [batch_size]
            
            # 使用softmax概率作为对齐分数
            probs_t = torch.softmax(output_t, dim=1)
            if label_t.dtype == torch.long:
                prob_t = probs_t.gather(1, label_t.unsqueeze(1)).squeeze(1)
            else:
                prob_t = torch.max(probs_t, dim=1)[0]
            
            alignment_scores.append(prob_t)
        
        return torch.stack(alignment_scores, dim=1)  # [batch_size, seq_len]

