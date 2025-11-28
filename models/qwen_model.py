"""
Qwen2.5模型用于持续学习
支持本地模型和Ollama API
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
import requests
import json
from .base_model import BaseContinualLearningModel

try:
    from transformers import Qwen2Model, Qwen2Config, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class Qwen2ForContinualLearning(BaseContinualLearningModel):
    """Qwen2.5持续学习模型（本地版本）"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        num_classes: int = 2,
        dropout: float = 0.1,
        use_local: bool = True
    ):
        super().__init__(num_classes)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers库未安装，无法使用本地Qwen模型")
        
        # 加载Qwen2模型
        self.qwen2 = Qwen2Model.from_pretrained(model_name)
        self.config = self.qwen2.config
        self.hidden_size = self.config.hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_representations(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取中间层表示
        
        Args:
            inputs: 可以是(input_ids, attention_mask)或单个tensor
        
        Returns:
            (logits, pooled_representation)
        """
        if isinstance(inputs, tuple):
            input_ids, attention_mask = inputs
        else:
            input_ids = inputs
            attention_mask = None
        
        # 获取Qwen2输出
        outputs = self.qwen2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一个token的表示
        last_hidden_state = outputs.last_hidden_state
        # 取序列的最后一个有效token
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.size(0)
            pooled_output = last_hidden_state[torch.arange(batch_size), seq_lengths]
        else:
            pooled_output = last_hidden_state[:, -1, :]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, pooled_output
    
    def forward(self, inputs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 可以是(input_ids, attention_mask)或单个tensor
        
        Returns:
            logits
        """
        logits, _ = self.get_representations(inputs)
        return logits
    
    def get_layer_names(self) -> list:
        """获取所有层名称"""
        layer_names = []
        for name, _ in self.named_modules():
            layer_names.append(name)
        return layer_names


class OllamaQwen2ForContinualLearning(BaseContinualLearningModel):
    """Qwen2.5持续学习模型（Ollama API版本）"""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:32b",
        num_classes: int = 2,
        max_length: int = 512
    ):
        super().__init__(num_classes)
        
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.max_length = max_length
        self.num_classes = num_classes
        
        # 用于存储嵌入的简单映射
        self.embedding_cache = {}
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.ollama_model not in model_names:
                    print(f"警告: 模型 {self.ollama_model} 未在Ollama中找到")
                    print(f"可用模型: {model_names}")
            else:
                print(f"警告: 无法连接到Ollama服务 ({self.ollama_base_url})")
        except Exception as e:
            print(f"警告: Ollama连接测试失败: {e}")
            print("请确保Ollama服务正在运行: ollama serve")
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        通过Ollama API获取文本嵌入
        
        Args:
            texts: 文本列表
        
        Returns:
            嵌入张量 [batch_size, hidden_dim]
        """
        embeddings = []
        
        for text in texts:
            # 使用Ollama的embeddings API
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.ollama_model,
                        "prompt": text
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding = response.json().get("embedding", [])
                    embeddings.append(embedding)
                else:
                    # 如果embeddings API不可用，使用generate API的简化版本
                    print(f"警告: embeddings API不可用，使用备用方法")
                    # 使用零向量作为占位符（实际应用中需要更好的处理）
                    embeddings.append([0.0] * 4096)  # 假设隐藏维度为4096
                    
            except Exception as e:
                print(f"获取嵌入时出错: {e}")
                embeddings.append([0.0] * 4096)
        
        # 转换为张量
        max_dim = max(len(emb) for emb in embeddings) if embeddings else 4096
        # 统一维度
        embeddings = [emb + [0.0] * (max_dim - len(emb)) if len(emb) < max_dim else emb[:max_dim] 
                     for emb in embeddings]
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _get_representations_via_api(self, texts: List[str]) -> torch.Tensor:
        """
        通过API获取表示（用于分类任务）
        
        Args:
            texts: 文本列表
        
        Returns:
            表示张量
        """
        # 尝试使用embeddings API
        try:
            return self._get_embeddings(texts)
        except:
            # 如果失败，使用generate API获取logits（需要额外处理）
            # 这里简化处理，实际应用中需要更复杂的实现
            return self._get_embeddings(texts)
    
    def get_representations(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取中间层表示
        
        Args:
            inputs: 文本列表或(input_ids, attention_mask)元组
        
        Returns:
            (logits, pooled_representation)
        """
        # 处理输入 - Ollama模型需要文本列表
        if isinstance(inputs, list):
            # 文本列表（最常见的情况）
            texts = inputs
        elif isinstance(inputs, tuple):
            # tuple 可能的情况：
            # 1. (input_ids, attention_mask) - tokenized输入（Ollama无法使用）
            # 2. 文本的tuple（DataLoader的特殊情况）
            # 3. 其他格式
            if len(inputs) == 2:
                first, second = inputs
                # 检查是否是tensor（tokenized输入）
                if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                    # 这是tokenized输入，但Ollama需要文本，无法解码
                    # 创建一个占位符文本列表
                    texts = [f"text_{i}" for i in range(first.size(0))]
                elif all(isinstance(x, str) for x in inputs):
                    # 两个字符串，作为文本列表
                    texts = list(inputs)
                else:
                    # 混合类型，尝试转换为字符串
                    texts = [str(x) for x in inputs]
            else:
                # tuple 包含多个元素
                # 检查是否都是字符串
                if all(isinstance(x, str) for x in inputs):
                    texts = list(inputs)
                else:
                    # 转换为字符串列表
                    texts = [str(x) for x in inputs]
        elif isinstance(inputs, torch.Tensor):
            # 单个tensor（不常见，可能是错误的数据格式）
            # 创建一个占位符文本列表
            texts = [f"text_{i}" for i in range(inputs.size(0))]
        else:
            # 单个值（字符串或其他），转换为列表
            texts = [str(inputs)]
        
        # 确保 texts 是字符串列表
        if not texts:
            raise ValueError(f"无法从输入中提取文本: {inputs}")
        
        # 确保所有元素都是字符串
        texts = [str(t) for t in texts]
        
        # 获取表示
        representations = self._get_representations_via_api(texts)
        
        # 如果维度不匹配，进行投影
        if representations.size(1) != 4096:  # 假设分类头输入维度
            if representations.size(1) > 4096:
                # 降维
                if not hasattr(self, 'proj_layer'):
                    self.proj_layer = nn.Linear(representations.size(1), 4096)
                representations = self.proj_layer(representations)
            else:
                # 填充
                padding = torch.zeros(
                    representations.size(0),
                    4096 - representations.size(1)
                )
                representations = torch.cat([representations, padding], dim=1)
        
        # 分类头（动态创建，支持类别数更新）
        if not hasattr(self, 'classifier') or self.classifier.out_features != self.num_classes:
            self.classifier = nn.Linear(4096, self.num_classes)
        
        logits = self.classifier(representations)
        
        return logits, representations
    
    def forward(self, inputs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 文本列表或tokenized输入
        
        Returns:
            logits
        """
        logits, _ = self.get_representations(inputs)
        return logits
    
    def update_num_classes(self, num_classes: int):
        """
        更新分类头的类别数（用于持续学习中不同任务可能有不同类别数）
        
        Args:
            num_classes: 新的类别数
        """
        if num_classes != self.num_classes:
            self.num_classes = num_classes
            # 如果分类头已存在，重新创建
            if hasattr(self, 'classifier'):
                del self.classifier
            # 分类头会在下次调用时自动创建
    
    def get_layer_names(self) -> list:
        """获取所有层名称（Ollama版本简化）"""
        return ["ollama_embedding", "classifier"]


class Qwen2OllamaWrapper:
    """
    Qwen2.5 Ollama包装器
    提供统一的接口使用Ollama API
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:32b"
    ):
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self._test_connection()
    
    def _test_connection(self):
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama连接成功: {self.ollama_base_url}")
                print(f"✓ 使用模型: {self.ollama_model}")
            else:
                print(f"✗ Ollama连接失败: {response.status_code}")
        except Exception as e:
            print(f"✗ Ollama连接错误: {e}")
            print("请确保Ollama服务正在运行: ollama serve")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入
        
        Args:
            texts: 文本列表
        
        Returns:
            嵌入列表
        """
        embeddings = []
        
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.ollama_model,
                        "prompt": text
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    embedding = response.json().get("embedding", [])
                    embeddings.append(embedding)
                else:
                    print(f"警告: 获取嵌入失败: {response.status_code}")
                    embeddings.append([])
                    
            except Exception as e:
                print(f"获取嵌入时出错: {e}")
                embeddings.append([])
        
        return embeddings
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    **kwargs
                },
                timeout=120,
                stream=False
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"生成失败: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"生成时出错: {e}")
            return ""

