"""
实验1：虚假遗忘识别验证

实验目标：
1. 验证提出的虚假遗忘识别指标和分析工具的有效性
2. 证明能够准确区分虚假遗忘和真实遗忘

实验组：
- 组1：基线对照组（无缓解策略）
- 组2：虚假遗忘诱导组（冻结底层30%）
- 组3：真实遗忘诱导组（高强度训练，最小化旧任务数据）
- 组4：混合遗忘组
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import model_config, dataset_config, experiment_config
from data.datasets import (
    load_clinc150, load_20newsgroups, load_split_mnist,
    load_permuted_mnist, create_dataloader
)
from metrics.evaluation import Evaluator
from metrics.alignment_score import AlignmentScore
from metrics.reversibility import ReversibilityScore

# 导入Qwen模型（论文实验使用）
try:
    from models.qwen_model import (
        Qwen2ForContinualLearning,
        OllamaQwen2ForContinualLearning
    )
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("错误: Qwen模型不可用，请安装相关依赖")
    raise

def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_task(
    model: nn.Module,
    train_loader,
    num_epochs: int,
    device: str,
    optimizer,
    criterion,
    task_id: int
):
    """训练单个任务"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Task {task_id}, Epoch {epoch+1}"):
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Task {task_id}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def run_experiment_group(
    group_name: str,
    tasks: dict,
    model_type: str = "ollama_qwen2.5",
    freeze_ratio: float = None,
    high_intensity: bool = False,
    device: str = "cuda"
):
    """
    运行实验组
    
    Args:
        group_name: 实验组名称
        tasks: 任务字典
        model_type: 模型类型
        freeze_ratio: 冻结比例（None表示不冻结）
        high_intensity: 是否高强度训练（诱导真实遗忘）
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"运行实验组: {group_name}")
    print(f"{'='*60}\n")
    
    set_seed(experiment_config.seed)
    
    # 加载模型（只支持Qwen模型）
    tokenizer = None  # Ollama模型不需要tokenizer
    
    if model_type == "qwen2.5" and QWEN_AVAILABLE:
        # 使用本地Qwen2.5模型
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        model = Qwen2ForContinualLearning(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            num_classes=2
        ).to(device)
    elif model_type == "ollama_qwen2.5" and QWEN_AVAILABLE:
        # 使用Ollama Qwen2.5模型
        model = OllamaQwen2ForContinualLearning(
            ollama_base_url=model_config.ollama_base_url,
            ollama_model=model_config.ollama_model,
            num_classes=2
        )
        tokenizer = None  # Ollama版本不需要tokenizer
        print(f"使用Ollama模型: {model_config.ollama_model}")
        print(f"Ollama服务地址: {model_config.ollama_base_url}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 应用冻结策略
    if freeze_ratio is not None:
        print(f"冻结底层 {freeze_ratio*100}% 的层")
        model.freeze_bottom_layers(ratio=freeze_ratio)
    
    # 初始化评测器
    evaluator = Evaluator(model, device)
    alignment_scorer = AlignmentScore(model, device)
    reversibility_scorer = ReversibilityScore(device)
    
    # 存储结果
    results = {
        'group_name': group_name,
        'task_results': {},
        'identification_results': {}
    }
    
    # 持续学习循环
    num_tasks = len(tasks)
    task_performances = {}  # {task_id: [performance_after_each_task]}
    
    for current_task_id in range(num_tasks):
        print(f"\n训练任务 {current_task_id + 1}/{num_tasks}")
        
        # 获取当前任务数据
        task_data = tasks[current_task_id]
        train_texts, train_labels = task_data['train']
        test_texts, test_labels = task_data['test']
        
        # 创建数据加载器
        train_loader = create_dataloader(
            train_texts, train_labels, current_task_id,
            batch_size=model_config.batch_size,
            shuffle=True,
            tokenizer=tokenizer
        )
        test_loader = create_dataloader(
            test_texts, test_labels, current_task_id,
            batch_size=model_config.batch_size,
            shuffle=False,
            tokenizer=tokenizer
        )
        
        # 训练配置
        num_epochs = 10 if high_intensity else model_config.num_epochs
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # 训练前评估
        eval_result_before = evaluator.evaluate_task(
            current_task_id, test_loader, current_epoch=0
        )
        performance_before = eval_result_before['accuracy']
        
        # 训练任务
        train_task(
            model, train_loader, num_epochs, device,
            optimizer, criterion, current_task_id
        )
        
        # 训练后评估
        eval_result_after = evaluator.evaluate_task(
            current_task_id, test_loader, current_epoch=num_epochs
        )
        performance_after = eval_result_after['accuracy']
        
        # 评估所有已学任务
        task_performances[current_task_id] = [performance_after]
        for prev_task_id in range(current_task_id):
            prev_test_loader = create_dataloader(
                tasks[prev_task_id]['test'][0],
                tasks[prev_task_id]['test'][1],
                prev_task_id,
                batch_size=model_config.batch_size,
                shuffle=False,
                tokenizer=tokenizer
            )
            prev_eval = evaluator.evaluate_task(
                prev_task_id, prev_test_loader, current_epoch=current_task_id
            )
            task_performances[prev_task_id].append(prev_eval['accuracy'])
        
        # 计算可逆性
        if current_task_id in evaluator.task_representations:
            reps = evaluator.task_representations[current_task_id]
            if len(reps) >= 2:
                epochs = sorted(reps.keys())
                rep_before = reps[epochs[0]]
                rep_after = reps[epochs[-1]]
                
                is_reversible, reversibility_score = reversibility_scorer.predict_reversibility(
                    rep_before, rep_after,
                    performance_before, performance_after
                )
                
                results['identification_results'][current_task_id] = {
                    'reversibility_score': reversibility_score,
                    'is_reversible': is_reversible,
                    'performance_before': performance_before,
                    'performance_after': performance_after
                }
        
        print(f"任务 {current_task_id} 性能: {performance_after:.4f}")
    
    # 计算综合指标
    metrics = evaluator.get_comprehensive_metrics()
    results['metrics'] = metrics
    
    # 保存结果
    output_file = os.path.join(
        experiment_config.output_dir,
        f"{group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    return results

def main(device: str = None):
    """主函数"""
    print("="*60)
    print("实验1：虚假遗忘识别验证")
    print("="*60)
    
    # 确定使用的设备
    if device is None:
        device = experiment_config.device
    
    # 检查CUDA是否可用
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        device = "cpu"
    elif device == "cuda" and torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 加载数据集（以CLINC-150为例）
    print("\n加载数据集...")
    tasks = load_clinc150(
        data_dir=dataset_config.data_dir,
        num_tasks=min(5, dataset_config.num_tasks)  # 简化处理，只使用5个任务
    )
    
    if tasks is None:
        print("无法加载CLINC-150，使用20 Newsgroups")
        tasks = load_20newsgroups(
            data_dir=dataset_config.data_dir,
            num_tasks=5
        )
    
    if tasks is None:
        print("无法加载数据集，退出")
        return
    
    print(f"加载了 {len(tasks)} 个任务")
    
    # 运行各个实验组
    all_results = {}
    
    # 组1：基线对照组
    results1 = run_experiment_group(
        "baseline_control",
        tasks,
        model_type="bert",
        freeze_ratio=None,
        high_intensity=False,
        device=device
    )
    all_results['baseline'] = results1
    
    # 组2：虚假遗忘诱导组
    results2 = run_experiment_group(
        "spurious_forgetting_induced",
        tasks,
        model_type="bert",
        freeze_ratio=0.3,
        high_intensity=False,
        device=device
    )
    all_results['spurious'] = results2
    
    # 组3：真实遗忘诱导组
    results3 = run_experiment_group(
        "true_forgetting_induced",
        tasks,
        model_type="bert",
        freeze_ratio=None,
        high_intensity=True,
        device=device
    )
    all_results['true'] = results3
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    for group_name, result in all_results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n{group_name}:")
            print(f"  平均准确率: {metrics.get('average_accuracy', 0):.4f}")
            print(f"  后向迁移: {metrics.get('backward_transfer', 0):.4f}")
            if 'identification_results' in result:
                rev_scores = [r['reversibility_score'] 
                             for r in result['identification_results'].values()]
                if rev_scores:
                    print(f"  平均可逆性分数: {np.mean(rev_scores):.4f}")

if __name__ == "__main__":
    main()

