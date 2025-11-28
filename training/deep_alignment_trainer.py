"""
深层对齐训练模块
实现论文中的Deep Alignment Training方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

class DeepAlignmentTrainer:
    """
    深层对齐训练器
    实现论文Algorithm 1: Deep Alignment Training
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        position_weight_alpha: float = 0.5,
        alignment_regularization_lambda: float = 0.2,
        deep_alignment_threshold: float = 0.7,
        target_alignment_depth: int = 10
    ):
        """
        初始化深层对齐训练器
        
        Args:
            model: 要训练的模型
            device: 设备
            learning_rate: 学习率
            position_weight_alpha: 位置权重因子α（论文中α=0.5）
            alignment_regularization_lambda: 对齐正则化系数λ（论文中0.1-0.3）
            deep_alignment_threshold: 深层对齐阈值τ_deep
            target_alignment_depth: 目标对齐深度（论文中D > 10）
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.alpha = position_weight_alpha
        self.lambda_reg = alignment_regularization_lambda
        self.tau_deep = deep_alignment_threshold
        self.target_depth = target_alignment_depth
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        
    def compute_position_weights(self, sequence_length: int) -> torch.Tensor:
        """
        计算位置权重
        w_t = 1 + α * t/T
        
        Args:
            sequence_length: 序列长度T
            
        Returns:
            位置权重张量 [T]
        """
        t = torch.arange(1, sequence_length + 1, dtype=torch.float32, device=self.device)
        weights = 1 + self.alpha * t / sequence_length
        return weights
    
    def compute_token_alignment_scores(
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
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # output_weights: [num_classes, hidden_dim]
        # outputs: [batch_size, seq_len, num_classes]
        outputs = torch.matmul(hidden_states, output_weights.t())
        
        # 处理标签
        if true_labels.dim() == 1:
            # 如果是单标签，扩展到序列长度
            true_labels = true_labels.unsqueeze(1).expand(-1, seq_len)
        
        # 计算每个位置的余弦相似度
        alignment_scores = []
        for t in range(seq_len):
            # 获取位置t的输出和标签
            output_t = outputs[:, t, :]  # [batch_size, num_classes]
            label_t = true_labels[:, t]  # [batch_size]
            
            # 获取对应类别的logit
            if label_t.dtype == torch.long:
                # 分类任务：获取对应类别的logit
                logit_t = output_t.gather(1, label_t.unsqueeze(1)).squeeze(1)
                # 归一化到[0, 1]（使用softmax）
                probs_t = F.softmax(output_t, dim=1)
                prob_t = probs_t.gather(1, label_t.unsqueeze(1)).squeeze(1)
                alignment_scores.append(prob_t)
            else:
                # 回归任务：使用余弦相似度
                # 这里简化处理
                alignment_scores.append(torch.ones(batch_size, device=self.device))
        
        return torch.stack(alignment_scores, dim=1)  # [batch_size, seq_len]
    
    def compute_alignment_regularization(
        self,
        alignment_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对齐正则化项
        R_align = λ * Σ_t ||A_t - A_{t+1}||^2
        
        Args:
            alignment_scores: 对齐分数 [batch_size, seq_len]
            
        Returns:
            正则化损失
        """
        if alignment_scores.size(1) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 计算相邻位置的对齐分数差
        diff = alignment_scores[:, 1:] - alignment_scores[:, :-1]  # [batch_size, seq_len-1]
        
        # 计算L2范数
        reg_loss = self.lambda_reg * torch.mean(diff ** 2)
        
        return reg_loss
    
    def compute_deep_alignment_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        output_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算深层对齐损失
        L_deep = Σ_t w_t * l_t + R_align
        
        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes] 或 [batch_size, num_classes]
            labels: 真实标签 [batch_size, seq_len] 或 [batch_size]
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_dim]
            output_weights: 输出层权重 [num_classes, hidden_dim]
            
        Returns:
            (总损失, 损失字典)
        """
        batch_size = logits.size(0)
        
        # 处理logits和labels的维度
        if logits.dim() == 2:
            # [batch_size, num_classes] -> [batch_size, 1, num_classes]
            logits = logits.unsqueeze(1)
            seq_len = 1
        else:
            seq_len = logits.size(1)
        
        if labels.dim() == 1:
            # [batch_size] -> [batch_size, 1]
            labels = labels.unsqueeze(1)
        
        # 计算位置权重
        position_weights = self.compute_position_weights(seq_len)  # [seq_len]
        
        # 计算每个位置的损失
        losses_per_token = []
        for t in range(seq_len):
            if seq_len == 1:
                logit_t = logits.squeeze(1)  # [batch_size, num_classes]
                label_t = labels.squeeze(1)  # [batch_size]
            else:
                logit_t = logits[:, t, :]  # [batch_size, num_classes]
                label_t = labels[:, t]  # [batch_size]
            
            # 计算交叉熵损失
            loss_t = F.cross_entropy(logit_t, label_t, reduction='none')  # [batch_size]
            losses_per_token.append(loss_t)
        
        # 加权损失
        losses_tensor = torch.stack(losses_per_token, dim=1)  # [batch_size, seq_len]
        weighted_losses = losses_tensor * position_weights.unsqueeze(0)  # [batch_size, seq_len]
        deep_loss = torch.mean(weighted_losses)
        
        # 计算对齐正则化
        alignment_scores = self.compute_token_alignment_scores(
            hidden_states, output_weights, labels
        )
        reg_loss = self.compute_alignment_regularization(alignment_scores)
        
        # 总损失
        total_loss = deep_loss + reg_loss
        
        loss_dict = {
            'deep_loss': deep_loss.item(),
            'regularization_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def compute_alignment_depth(
        self,
        alignment_scores: torch.Tensor
    ) -> int:
        """
        计算对齐深度
        D(θ, T) = max{k : A_t(θ, T) ≥ τ_deep for all t ≤ k}
        
        Args:
            alignment_scores: 对齐分数 [batch_size, seq_len]
            
        Returns:
            对齐深度D
        """
        # 计算每个位置的平均对齐分数
        avg_scores = torch.mean(alignment_scores, dim=0)  # [seq_len]
        
        # 找到满足阈值的最长连续序列
        depth = 0
        for t in range(avg_scores.size(0)):
            if avg_scores[t].item() >= self.tau_deep:
                depth = t + 1
            else:
                break
        
        return depth
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
        max_epochs: int
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            max_epochs: 最大epoch数
            
        Returns:
            训练统计信息
        """
        self.model.train()
        total_loss = 0.0
        total_deep_loss = 0.0
        total_reg_loss = 0.0
        alignment_depths = []
        
        # 获取输出层权重
        output_weights = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' in name.lower():
                output_weights = module.weight.data
                break
        
        if output_weights is None:
            # 如果没有找到，使用最后一个Linear层
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    output_weights = module.weight.data
                    break
        
        if output_weights is None:
            raise ValueError("无法找到输出层权重")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")):
            # 处理输入
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 获取隐藏状态（需要模型支持）
            if hasattr(self.model, 'get_hidden_states'):
                hidden_states = self.model.get_hidden_states(inputs)
                logits = self.model(inputs)
            else:
                # 简化处理：使用模型的输出
                logits = self.model(inputs)
                # 假设隐藏状态与logits相关（实际需要从模型中获取）
                # 这里使用一个占位符，实际实现需要根据具体模型调整
                batch_size = logits.size(0)
                hidden_dim = output_weights.size(1)
                if logits.dim() == 2:
                    seq_len = 1
                else:
                    seq_len = logits.size(1)
                hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            
            # 计算损失
            loss, loss_dict = self.compute_deep_alignment_loss(
                logits, labels, hidden_states, output_weights
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss']
            total_deep_loss += loss_dict['deep_loss']
            total_reg_loss += loss_dict['regularization_loss']
            
            # 计算对齐深度（每10个batch计算一次）
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    alignment_scores = self.compute_token_alignment_scores(
                        hidden_states, output_weights, labels
                    )
                    depth = self.compute_alignment_depth(alignment_scores)
                    alignment_depths.append(depth)
        
        avg_loss = total_loss / len(dataloader)
        avg_deep_loss = total_deep_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        avg_depth = np.mean(alignment_depths) if alignment_depths else 0
        
        return {
            'loss': avg_loss,
            'deep_loss': avg_deep_loss,
            'regularization_loss': avg_reg_loss,
            'alignment_depth': avg_depth
        }
    
    def train(
        self,
        dataloader,
        max_epochs: int = 3,
        early_stop_depth: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        训练模型直到达到深层对齐
        
        Args:
            dataloader: 数据加载器
            max_epochs: 最大epoch数
            early_stop_depth: 早停深度（如果达到此深度则停止）
            
        Returns:
            训练历史
        """
        history = {
            'loss': [],
            'deep_loss': [],
            'regularization_loss': [],
            'alignment_depth': []
        }
        
        for epoch in range(max_epochs):
            stats = self.train_epoch(dataloader, epoch, max_epochs)
            
            history['loss'].append(stats['loss'])
            history['deep_loss'].append(stats['deep_loss'])
            history['regularization_loss'].append(stats['regularization_loss'])
            history['alignment_depth'].append(stats['alignment_depth'])
            
            print(f"Epoch {epoch+1}: Loss={stats['loss']:.4f}, "
                  f"Deep Loss={stats['deep_loss']:.4f}, "
                  f"Reg Loss={stats['regularization_loss']:.4f}, "
                  f"Alignment Depth={stats['alignment_depth']:.2f}")
            
            # 早停检查
            if early_stop_depth is not None:
                if stats['alignment_depth'] >= early_stop_depth:
                    print(f"达到目标对齐深度 {early_stop_depth}，提前停止训练")
                    break
        
        return history

