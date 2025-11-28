"""
基础模型类
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BaseContinualLearningModel(nn.Module):
    """持续学习模型基类"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.frozen_layers = set()
    
    def freeze_layers(self, layer_names: list, freeze: bool = True):
        """冻结或解冻指定层"""
        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = not freeze
                    if freeze:
                        self.frozen_layers.add(name)
                    else:
                        self.frozen_layers.discard(name)
    
    def freeze_bottom_layers(self, ratio: float = 0.3):
        """冻结底层一定比例的层"""
        all_layers = list(self.named_parameters())
        num_layers = len(all_layers)
        num_freeze = int(num_layers * ratio)
        
        layer_names = [name for name, _ in all_layers[:num_freeze]]
        self.freeze_layers(layer_names, freeze=True)
    
    def freeze_top_layers(self, ratio: float = 0.3):
        """冻结顶层一定比例的层"""
        all_layers = list(self.named_parameters())
        num_layers = len(all_layers)
        num_freeze = int(num_layers * ratio)
        
        layer_names = [name for name, _ in all_layers[-num_freeze:]]
        self.freeze_layers(layer_names, freeze=True)
    
    def get_representations(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取中间层表示（子类需要实现）"""
        raise NotImplementedError
    
    def forward(self, inputs) -> torch.Tensor:
        """前向传播（子类需要实现）"""
        raise NotImplementedError

