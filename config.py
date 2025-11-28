"""
配置文件：定义实验参数和路径
"""
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "qwen2.5"  # 默认模型名称（用于兼容性）
    model_type: str = "ollama_qwen2.5"  # qwen2.5, ollama_qwen2.5（论文实验使用Ollama Qwen）
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    freeze_bottom_layers: Optional[float] = None  # 冻结底层比例，如0.3表示30%
    freeze_top_layers: Optional[float] = None     # 冻结顶层比例
    
    # Ollama配置 - 论文中使用的4个模型
    ollama_base_url: str = "http://localhost:11434"  # Ollama服务地址
    ollama_model: str = "qwen2.5:3b"  # Ollama模型名称（默认）
    use_ollama: bool = True  # 是否使用Ollama（论文实验默认使用）
    
    # 论文中使用的4个Ollama模型列表（对应论文Section 5.1）
    ollama_models: List[str] = None  # 将在初始化时设置
    
    def __post_init__(self):
        """初始化后处理"""
        if self.ollama_models is None:
            # 论文中的4个模型：Qwen3-1.7B, Qwen2.5-3B, Qwen3-4B, Qwen2.5-32B
            # 对应论文实验设置（Section 5.1: Experimental Setup）
            self.ollama_models = [
                "qwen3:1.7b",      # Qwen3-1.7B (1.7B parameters)
                "qwen2.5:3b",    # Qwen2.5-3B (3B parameters)
                "qwen3:4b",      # Qwen3-4B (4B parameters)
                "qwen2.5:32b"    # Qwen2.5-32B (32B parameters)
            ]

@dataclass
class DatasetConfig:
    """数据集配置"""
    dataset_name: str = "clinc150"  # clinc150, 20newsgroups, split_mnist, permuted_mnist, split_cifar10
    data_dir: str = "./data"
    num_tasks: int = 15
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "spurious_forgetting_identification"
    output_dir: str = "./results"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "cuda"  # cuda or cpu
    num_runs: int = 5  # 每个实验运行次数
    
    # 虚假遗忘识别相关 - 论文中的阈值
    alignment_threshold: float = 0.7  # τ_align: 对齐度阈值
    reversibility_threshold: float = 0.6  # τ_R: 可逆性阈值
    spurious_forgetting_threshold: float = 0.6  # τ_S: 虚假遗忘分数阈值
    deep_alignment_threshold: float = 0.7  # τ_deep: 深层对齐阈值
    alignment_depth_threshold: int = 5  # 浅层对齐的深度阈值（D ≤ 5）
    deep_alignment_depth: int = 10  # 深层对齐的深度阈值（D > 10）
    
    # 深层对齐训练相关（对应论文Section 4.3: Deep Alignment Training）
    use_deep_alignment_training: bool = True  # 是否使用深层对齐训练（论文实验默认启用）
    position_weight_alpha: float = 0.5  # 位置权重因子α（论文中α=0.5）
    alignment_regularization_lambda: float = 0.2  # 对齐正则化系数λ（论文中0.1-0.3，默认0.2）
    
    # 缓解策略相关（对应论文Section 4.4: Adaptive Mitigation Strategies）
    use_adaptive_freezing: bool = True  # 使用自适应冻结（论文实验默认启用）
    use_alignment_repair: bool = True  # 使用选择性对齐修复（论文实验默认启用）
    use_hybrid_strategy: bool = True  # 使用混合策略（论文实验默认启用，对应Algorithm 2）
    freeze_threshold: float = 0.7  # τ_freeze: 冻结阈值
    
    # 选择性修复相关
    repair_samples: int = 75  # 修复时使用的样本数（50-100）
    repair_learning_rate: float = 1e-4  # 修复学习率
    repair_max_epochs: int = 3  # 修复最大轮数
    repair_alignment_target: float = 0.85  # 修复目标对齐度
    
    # 基线方法
    use_ewc: bool = False
    use_si: bool = False
    use_replay: bool = False
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.2  # 经验回放比例（20%）

@dataclass
class AnalysisConfig:
    """分析工具配置"""
    track_representations: bool = True
    track_frequency: int = 10  # 每N个epoch记录一次
    use_cka: bool = True
    use_pca: bool = True
    visualization: bool = True

# 全局配置实例
model_config = ModelConfig()
dataset_config = DatasetConfig()
experiment_config = ExperimentConfig()
analysis_config = AnalysisConfig()

# 创建必要的目录
os.makedirs(experiment_config.output_dir, exist_ok=True)
os.makedirs(experiment_config.log_dir, exist_ok=True)
os.makedirs(dataset_config.data_dir, exist_ok=True)

