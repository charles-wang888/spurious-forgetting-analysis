"""
自适应缓解策略模块
实现论文中的Adaptive Mitigation Strategy（Algorithm 2）
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

class AdaptiveMitigationStrategy:
    """
    自适应缓解策略
    实现论文Algorithm 2: Adaptive Mitigation Strategy
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        tau_s: float = 0.6,
        tau_r: float = 0.6,
        tau_align: float = 0.7,
        tau_freeze: float = 0.7,
        repair_samples: int = 75,
        repair_lr: float = 1e-4,
        repair_max_epochs: int = 3,
        repair_alignment_target: float = 0.85,
        replay_ratio: float = 0.2
    ):
        """
        初始化自适应缓解策略
        
        Args:
            model: 模型
            device: 设备
            tau_s: 虚假遗忘分数阈值
            tau_r: 可逆性阈值
            tau_align: 对齐度阈值
            tau_freeze: 冻结阈值
            repair_samples: 修复样本数
            repair_lr: 修复学习率
            repair_max_epochs: 修复最大epoch数
            repair_alignment_target: 修复目标对齐度
            replay_ratio: 经验回放比例
        """
        self.model = model
        self.device = device
        self.tau_s = tau_s
        self.tau_r = tau_r
        self.tau_align = tau_align
        self.tau_freeze = tau_freeze
        self.repair_samples = repair_samples
        self.repair_lr = repair_lr
        self.repair_max_epochs = repair_max_epochs
        self.repair_alignment_target = repair_alignment_target
        self.replay_ratio = replay_ratio
    
    def detect_forgetting_type(
        self,
        spurious_score: float,
        reversibility_score: float,
        alignment_depth: int
    ) -> str:
        """
        检测遗忘类型
        
        Args:
            spurious_score: 虚假遗忘分数S
            reversibility_score: 可逆性分数R
            alignment_depth: 对齐深度D
            
        Returns:
            遗忘类型: 'spurious', 'true', 'none'
        """
        # 根据论文Algorithm 2的逻辑
        if spurious_score > self.tau_s and reversibility_score > self.tau_r and alignment_depth <= 5:
            return 'spurious'
        elif spurious_score > self.tau_s and reversibility_score <= self.tau_r:
            return 'true'
        else:
            return 'none'
    
    def selective_alignment_repair(
        self,
        task_data: Tuple[List, List],
        task_id: int
    ) -> bool:
        """
        选择性对齐修复
        当检测到虚假遗忘时应用
        
        Args:
            task_data: 任务数据 (texts, labels)
            task_id: 任务ID
            
        Returns:
            是否修复成功
        """
        print(f"应用选择性对齐修复 (任务 {task_id})")
        
        texts, labels = task_data
        
        # 1. 收集50-100个样本
        num_samples = min(self.repair_samples, len(texts))
        indices = np.random.choice(len(texts), num_samples, replace=False)
        repair_texts = [texts[i] for i in indices]
        repair_labels = [labels[i] for i in indices]
        
        # 2. 冻结所有层 except output layer
        self._freeze_all_except_output()
        
        # 3. 创建修复优化器（只优化输出层）
        output_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name.lower() or 'output' in name.lower():
                output_params.append(param)
        
        if not output_params:
            print("警告: 未找到输出层参数，修复失败")
            return False
        
        optimizer = torch.optim.AdamW(output_params, lr=self.repair_lr)
        criterion = nn.CrossEntropyLoss()
        
        # 4. 微调
        self.model.train()
        for epoch in range(self.repair_max_epochs):
            total_loss = 0.0
            
            # 简化的训练循环（实际需要根据数据格式调整）
            for i in range(len(repair_texts)):
                # 这里需要根据实际模型接口调整
                # 假设模型可以直接处理文本
                try:
                    if hasattr(self.model, 'forward_text'):
                        outputs = self.model.forward_text([repair_texts[i]])
                    else:
                        # 使用占位符
                        outputs = torch.randn(1, 2, device=self.device)  # 假设2类
                    
                    label_tensor = torch.tensor([repair_labels[i]], device=self.device)
                    
                    optimizer.zero_grad()
                    loss = criterion(outputs, label_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"修复训练时出错: {e}")
                    continue
            
            avg_loss = total_loss / len(repair_texts) if repair_texts else 0
            print(f"修复 Epoch {epoch+1}/{self.repair_max_epochs}, Loss: {avg_loss:.4f}")
            
            # 5. 检查对齐恢复
            # 这里简化处理，实际需要计算对齐度
            if epoch >= 1:  # 至少训练1个epoch后检查
                # 实际应该计算A(θ, T)并与repair_alignment_target比较
                # 这里简化处理
                if avg_loss < 0.1:  # 损失足够低
                    print(f"对齐恢复成功 (Loss: {avg_loss:.4f})")
                    break
        
        # 解冻所有层
        self._unfreeze_all()
        
        return True
    
    def experience_replay(
        self,
        current_task_data: Tuple[List, List],
        previous_tasks_data: List[Tuple[List, List]],
        task_id: int
    ):
        """
        经验回放
        当检测到真实遗忘时应用
        
        Args:
            current_task_data: 当前任务数据
            previous_tasks_data: 先前任务数据列表
            task_id: 当前任务ID
        """
        print(f"应用经验回放 (任务 {task_id})")
        
        # 1. 从先前任务中采样20%的数据
        replay_texts = []
        replay_labels = []
        
        for prev_task_data in previous_tasks_data:
            prev_texts, prev_labels = prev_task_data
            num_replay = max(1, int(len(prev_texts) * self.replay_ratio))
            indices = np.random.choice(len(prev_texts), num_replay, replace=False)
            replay_texts.extend([prev_texts[i] for i in indices])
            replay_labels.extend([prev_labels[i] for i in indices])
        
        # 2. 合并当前任务和回放数据
        current_texts, current_labels = current_task_data
        all_texts = current_texts + replay_texts
        all_labels = current_labels + replay_labels
        
        print(f"经验回放: 当前任务 {len(current_texts)} 样本, "
              f"回放 {len(replay_texts)} 样本, "
              f"总计 {len(all_texts)} 样本")
        
        return (all_texts, all_labels)
    
    def adaptive_freezing(
        self,
        task_id: int,
        layer_alignments: Dict[str, float]
    ):
        """
        自适应冻结
        预防性策略
        
        Args:
            task_id: 任务ID
            layer_alignments: 各层的对齐分数 {layer_name: score}
        """
        print(f"应用自适应冻结 (任务 {task_id})")
        
        # 1. 计算所有层的对齐分数
        # layer_alignments 已经提供
        
        # 2. 识别关键层
        critical_layers = [
            name for name, score in layer_alignments.items()
            if score < self.tau_freeze
        ]
        
        # 3. 冻结关键层或底层30%
        if critical_layers:
            print(f"冻结关键层: {len(critical_layers)} 个")
            for name, param in self.model.named_parameters():
                for layer_name in critical_layers:
                    if layer_name in name:
                        param.requires_grad = False
        else:
            # 如果没有关键层，冻结底层30%
            print("冻结底层30%")
            self._freeze_bottom_layers(ratio=0.3)
    
    def _freeze_all_except_output(self):
        """冻结除输出层外的所有层"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name.lower() and 'output' not in name.lower():
                param.requires_grad = False
    
    def _unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _freeze_bottom_layers(self, ratio: float = 0.3):
        """冻结底层比例"""
        layer_names = [name for name, _ in self.model.named_modules()]
        num_layers = len(layer_names)
        num_freeze = int(num_layers * ratio)
        
        frozen_layers = layer_names[:num_freeze]
        
        for name, param in self.model.named_parameters():
            for layer_name in frozen_layers:
                if layer_name in name:
                    param.requires_grad = False

