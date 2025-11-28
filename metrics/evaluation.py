"""
综合评测模块
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .alignment_score import AlignmentScore
from .reversibility import ReversibilityScore
from .forgetting_metrics import (
    compute_forgetting_measure,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_average_accuracy,
    compute_forgetting_rate
)

class Evaluator:
    """
    综合评测器
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.alignment_scorer = AlignmentScore(model, device)
        self.reversibility_scorer = ReversibilityScore(device)
        
        # 存储历史数据
        self.task_accuracies = {}  # {task_id: [acc_after_task_1, acc_after_task_2, ...]}
        self.task_representations = {}  # {task_id: {epoch: representation}}
        self.alignment_scores = {}  # {task_id: {epoch: alignment_score}}
        self.reversibility_scores = {}  # {task_id: reversibility_score}
    
    def evaluate_task(
        self,
        task_id: int,
        dataloader,
        current_epoch: int = 0
    ) -> Dict:
        """
        评估单个任务
        
        Args:
            task_id: 任务ID
            dataloader: 数据加载器
            current_epoch: 当前epoch
        
        Returns:
            评估结果字典
        """
        self.model.eval()
        correct = 0
        total = 0
        all_representations = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                
                # 处理 inputs 可能是 tuple 的情况（文本数据的 tokenizer 返回）
                if isinstance(inputs, tuple):
                    # 如果是 tuple，每个元素单独移动到设备
                    inputs = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs)
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                
                # 确保 labels 是 tensor
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                else:
                    labels = torch.tensor(labels).to(self.device)
                
                # 获取模型输出和表示
                if hasattr(self.model, 'get_representations'):
                    outputs, representations = self.model.get_representations(inputs)
                else:
                    outputs = self.model(inputs)
                    representations = None
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if representations is not None:
                    all_representations.append(representations.cpu())
                    all_labels.append(labels.cpu())
        
        accuracy = correct / total if total > 0 else 0.0
        
        # 更新历史记录
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = []
        self.task_accuracies[task_id].append(accuracy)
        
        # 存储表示
        if len(all_representations) > 0:
            if task_id not in self.task_representations:
                self.task_representations[task_id] = {}
            self.task_representations[task_id][current_epoch] = torch.cat(all_representations, dim=0)
        
        return {
            'task_id': task_id,
            'accuracy': accuracy,
            'total': total,
            'correct': correct
        }
    
    def compute_alignment_for_task(
        self,
        task_id: int,
        layer_names: List[str]
    ) -> Dict[str, float]:
        """
        计算任务的对齐度
        
        Args:
            task_id: 任务ID
            layer_names: 要分析的层名称列表
        
        Returns:
            各层的对齐度分数
        """
        # 这里需要实际运行一次前向传播来获取表示
        # 简化实现，实际使用时需要传入数据
        return {}
    
    def compute_reversibility_for_task(
        self,
        task_id: int,
        performance_before: float,
        performance_after: float
    ) -> Tuple[bool, float]:
        """
        计算任务的可逆性
        
        Args:
            task_id: 任务ID
            performance_before: 训练前的性能
            performance_after: 训练后的性能
        
        Returns:
            (is_reversible, reversibility_score)
        """
        if task_id not in self.task_representations:
            return False, 0.0
        
        representations = self.task_representations[task_id]
        if len(representations) < 2:
            return False, 0.0
        
        # 获取第一个和最后一个epoch的表示
        epochs = sorted(representations.keys())
        rep_before = representations[epochs[0]]
        rep_after = representations[epochs[-1]]
        
        is_reversible, score = self.reversibility_scorer.predict_reversibility(
            rep_before,
            rep_after,
            performance_before,
            performance_after
        )
        
        if task_id not in self.reversibility_scores:
            self.reversibility_scores[task_id] = score
        
        return is_reversible, score
    
    def get_comprehensive_metrics(self) -> Dict:
        """
        获取综合评测指标
        
        Returns:
            包含所有指标的字典
        """
        metrics = {
            'average_accuracy': compute_average_accuracy(self.task_accuracies),
            'backward_transfer': compute_backward_transfer(self.task_accuracies),
            'forward_transfer': compute_forward_transfer(self.task_accuracies),
            'forgetting_measure': compute_forgetting_measure(self.task_accuracies),
            'forgetting_rate': compute_forgetting_rate(self.task_accuracies),
            'reversibility_scores': self.reversibility_scores
        }
        
        return metrics
    
    def reset(self):
        """重置评测器"""
        self.task_accuracies = {}
        self.task_representations = {}
        self.alignment_scores = {}
        self.reversibility_scores = {}

