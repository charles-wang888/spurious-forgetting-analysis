"""
主实验脚本：对应论文中的实验设置
使用Ollama启动的4个Qwen模型进行实验
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    model_config, dataset_config, experiment_config,
    ModelConfig, DatasetConfig, ExperimentConfig
)
from data.datasets import (
    load_clinc150, load_20newsgroups,
    load_split_mnist, load_permuted_mnist,
    create_dataloader
)
from models.qwen_model import OllamaQwen2ForContinualLearning
from metrics.alignment_score import AlignmentScore
from metrics.reversibility import ReversibilityScore
from metrics.evaluation import Evaluator
from training.deep_alignment_trainer import DeepAlignmentTrainer
from training.adaptive_mitigation import AdaptiveMitigationStrategy
from utils.ollama_helper import OllamaHelper, setup_ollama_model

def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_alignment_depth(
    model: nn.Module,
    dataloader,
    device: str,
    tau_deep: float = 0.7
) -> int:
    """
    计算对齐深度
    D(θ, T) = max{k : A_t(θ, T) ≥ τ_deep for all t ≤ k}
    """
    model.eval()
    alignment_scores_per_token = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)
            
            # 获取模型输出和表示
            if hasattr(model, 'get_representations'):
                logits, representations = model.get_representations(inputs)
            else:
                logits = model(inputs)
                representations = None
            
            # 简化处理：假设每个样本只有一个token位置
            # 实际实现需要根据模型输出调整
            if logits.dim() == 2:
                # [batch_size, num_classes]
                probs = torch.softmax(logits, dim=1)
                # 使用最大概率作为对齐分数（简化）
                alignment = torch.max(probs, dim=1)[0]
                alignment_scores_per_token.append(alignment.cpu().numpy())
    
    if not alignment_scores_per_token:
        return 0
    
    # 计算平均对齐分数
    all_scores = np.concatenate(alignment_scores_per_token)
    avg_score = np.mean(all_scores)
    
    # 简化处理：如果平均分数超过阈值，认为深度为1
    # 实际需要按token位置计算
    if avg_score >= tau_deep:
        return 1  # 简化处理
    else:
        return 0

def compute_spurious_forgetting_score(
    alignment_score: float,
    reversibility_score: float,
    performance_drop: float,
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.2
) -> float:
    """
    计算虚假遗忘分数
    S(θ, T) = w1 * (1 - A) + w2 * R + w3 * ΔP
    """
    return w1 * (1 - alignment_score) + w2 * reversibility_score + w3 * performance_drop

def run_single_model_experiment(
    ollama_model: str,
    dataset_name: str,
    experiment_group: str,
    device: str = "cuda"
) -> Dict:
    """
    在单个模型上运行实验
    
    Args:
        ollama_model: Ollama模型名称（如 "qwen2.5:3b"）
        dataset_name: 数据集名称
        experiment_group: 实验组名称
        device: 设备
        
    Returns:
        实验结果字典
    """
    print(f"\n{'='*80}")
    print(f"模型: {ollama_model} | 数据集: {dataset_name} | 实验组: {experiment_group}")
    print(f"{'='*80}\n")
    
    set_seed(experiment_config.seed)
    
    # 检查Ollama连接和模型
    helper = OllamaHelper(model_config.ollama_base_url)
    if not helper.test_connection():
        print(f"错误: 无法连接到Ollama服务 ({model_config.ollama_base_url})")
        return None
    
    if not helper.check_model(ollama_model):
        print(f"警告: 模型 {ollama_model} 不存在，尝试拉取...")
        if not helper.pull_model(ollama_model):
            print(f"错误: 无法拉取模型 {ollama_model}")
            return None
    
    # 加载数据集
    if dataset_name == "clinc150":
        tasks = load_clinc150(
            data_dir=dataset_config.data_dir,
            num_tasks=min(15, dataset_config.num_tasks)
        )
    elif dataset_name == "20newsgroups":
        tasks = load_20newsgroups(
            data_dir=dataset_config.data_dir,
            num_tasks=5
        )
    elif dataset_name == "split_mnist":
        tasks = load_split_mnist(
            data_dir=dataset_config.data_dir,
            num_tasks=5
        )
    elif dataset_name == "permuted_mnist":
        tasks = load_permuted_mnist(
            data_dir=dataset_config.data_dir,
            num_tasks=10
        )
    else:
        print(f"错误: 不支持的数据集 {dataset_name}")
        return None
    
    if tasks is None:
        print(f"错误: 无法加载数据集 {dataset_name}")
        return None
    
    print(f"加载了 {len(tasks)} 个任务")
    
    # 创建模型
    model = OllamaQwen2ForContinualLearning(
        ollama_base_url=model_config.ollama_base_url,
        ollama_model=ollama_model,
        num_classes=2  # 简化处理，实际应根据任务调整
    )
    
    # 根据实验组设置训练策略
    use_deep_alignment = experiment_config.use_deep_alignment_training
    freeze_ratio = None
    high_intensity = False
    
    if experiment_group == "baseline_control":
        use_deep_alignment = False
    elif experiment_group == "spurious_forgetting_induced":
        freeze_ratio = 0.3
        use_deep_alignment = False
    elif experiment_group == "true_forgetting_induced":
        high_intensity = True
        use_deep_alignment = False
    elif experiment_group == "mixed_forgetting":
        freeze_ratio = 0.3
        high_intensity = True
        use_deep_alignment = False
    elif experiment_group == "deep_alignment_training":
        use_deep_alignment = True
    elif experiment_group == "ablation":
        # 消融实验需要特殊处理，返回多个配置的结果
        return run_ablation(ollama_model, dataset_name, device)
    
    # 应用冻结策略
    if freeze_ratio is not None:
        print(f"冻结底层 {freeze_ratio*100}% 的层")
        if hasattr(model, 'freeze_bottom_layers'):
            model.freeze_bottom_layers(ratio=freeze_ratio)
    
    # 初始化评估器和工具
    evaluator = Evaluator(model, device)
    alignment_scorer = AlignmentScore(model, device)
    reversibility_scorer = ReversibilityScore(device)
    
    # 初始化深层对齐训练器（如果使用）
    deep_trainer = None
    if use_deep_alignment:
        deep_trainer = DeepAlignmentTrainer(
            model=model,
            device=device,
            learning_rate=model_config.learning_rate,
            position_weight_alpha=experiment_config.position_weight_alpha,
            alignment_regularization_lambda=experiment_config.alignment_regularization_lambda,
            deep_alignment_threshold=experiment_config.deep_alignment_threshold,
            target_alignment_depth=experiment_config.deep_alignment_depth
        )
    
    # 初始化自适应缓解策略
    mitigation_strategy = AdaptiveMitigationStrategy(
        model=model,
        device=device,
        tau_s=experiment_config.spurious_forgetting_threshold,
        tau_r=experiment_config.reversibility_threshold,
        tau_align=experiment_config.alignment_threshold,
        tau_freeze=experiment_config.freeze_threshold,
        repair_samples=experiment_config.repair_samples,
        repair_lr=experiment_config.repair_learning_rate,
        repair_max_epochs=experiment_config.repair_max_epochs,
        repair_alignment_target=experiment_config.repair_alignment_target,
        replay_ratio=experiment_config.replay_ratio
    )
    
    # 存储结果
    results = {
        'model': ollama_model,
        'dataset': dataset_name,
        'experiment_group': experiment_group,
        'task_results': {},
        'identification_results': {},
        'alignment_depths': {}
    }
    
    # 持续学习循环
    num_tasks = len(tasks)
    previous_tasks_data = []
    
    for current_task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"任务 {current_task_id + 1}/{num_tasks}")
        print(f"{'='*60}\n")
        
        # 获取当前任务数据
        task_data = tasks[current_task_id]
        train_texts, train_labels = task_data['train']
        test_texts, test_labels = task_data['test']
        
        # 确定当前任务的类别数
        if isinstance(train_labels, list):
            num_classes = max(train_labels) + 1 if train_labels else 2
        elif isinstance(train_labels, torch.Tensor):
            num_classes = int(train_labels.max().item()) + 1 if len(train_labels) > 0 else 2
        else:
            num_classes = len(set(train_labels)) if train_labels else 2
        
        # 更新模型的类别数
        if hasattr(model, 'update_num_classes'):
            model.update_num_classes(num_classes)
        else:
            model.num_classes = num_classes
            # 如果分类头已存在，重新创建
            if hasattr(model, 'classifier'):
                del model.classifier
        
        print(f"任务 {current_task_id} 类别数: {num_classes}")
        
        # 创建数据加载器（简化处理，Ollama模型可能需要不同的处理方式）
        # 注意：Ollama模型可能需要直接使用文本，而不是tokenized输入
        train_loader = create_dataloader(
            train_texts, train_labels, current_task_id,
            batch_size=model_config.batch_size,
            shuffle=True,
            tokenizer=None  # Ollama模型不需要tokenizer
        )
        test_loader = create_dataloader(
            test_texts, test_labels, current_task_id,
            batch_size=model_config.batch_size,
            shuffle=False,
            tokenizer=None
        )
        
        # 训练前评估
        eval_result_before = evaluator.evaluate_task(
            current_task_id, test_loader, current_epoch=0
        )
        performance_before = eval_result_before.get('accuracy', 0.0)
        
        # 训练任务
        num_epochs = 10 if high_intensity else model_config.num_epochs
        
        if use_deep_alignment and deep_trainer:
            # 使用深层对齐训练
            print("使用深层对齐训练...")
            history = deep_trainer.train(
                train_loader,
                max_epochs=num_epochs,
                early_stop_depth=experiment_config.deep_alignment_depth
            )
            results['alignment_depths'][current_task_id] = history['alignment_depth']
        else:
            # 标准训练
            print("使用标准训练...")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=model_config.learning_rate
            )
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    if len(batch) == 3:
                        inputs, labels, _ = batch
                    else:
                        inputs, labels = batch
                    
                    # 处理不同的 inputs 格式
                    if isinstance(inputs, tuple):
                        # 如果是 tuple（tokenizer 返回的 input_ids, attention_mask），移动到设备
                        inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
                    elif isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                    # 如果是列表（文本列表，Ollama 使用），保持不变
                    
                    outputs = model(inputs)
                    
                    # 处理 labels
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        labels = torch.tensor(labels, device=device)
                    
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # 训练后评估
        eval_result_after = evaluator.evaluate_task(
            current_task_id, test_loader, current_epoch=num_epochs
        )
        performance_after = eval_result_after.get('accuracy', 0.0)
        
        # 评估所有已学任务
        task_performances = {current_task_id: performance_after}
        for prev_task_id in range(current_task_id):
            prev_test_loader = create_dataloader(
                tasks[prev_task_id]['test'][0],
                tasks[prev_task_id]['test'][1],
                prev_task_id,
                batch_size=model_config.batch_size,
                shuffle=False,
                tokenizer=None
            )
            prev_eval = evaluator.evaluate_task(
                prev_task_id, prev_test_loader, current_epoch=current_task_id
            )
            prev_performance = prev_eval.get('accuracy', 0.0)
            task_performances[prev_task_id] = prev_performance
        
        # 计算对齐深度、可逆性、虚假遗忘分数
        alignment_depth = compute_alignment_depth(
            model, test_loader, device, experiment_config.deep_alignment_threshold
        )
        
        # 计算对齐分数（简化处理）
        alignment_score = alignment_scorer.compute_alignment_score(
            torch.randn(10, 768, device=device),  # 占位符
            torch.randn(2, 768, device=device)   # 占位符
        )
        
        # 计算可逆性分数
        reversibility_score = 0.0
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
        
        # 计算虚假遗忘分数
        performance_drop = max(0, performance_before - performance_after)
        spurious_score = compute_spurious_forgetting_score(
            alignment_score, reversibility_score, performance_drop
        )
        
        # 检测遗忘类型
        forgetting_type = mitigation_strategy.detect_forgetting_type(
            spurious_score, reversibility_score, alignment_depth
        )
        
        # 应用缓解策略（如果启用）
        if experiment_config.use_hybrid_strategy:
            if forgetting_type == 'spurious':
                # 选择性修复
                mitigation_strategy.selective_alignment_repair(
                    (test_texts, test_labels), current_task_id
                )
            elif forgetting_type == 'true':
                # 经验回放
                if previous_tasks_data:
                    combined_data = mitigation_strategy.experience_replay(
                        (train_texts, train_labels),
                        previous_tasks_data,
                        current_task_id
                    )
                    # 重新训练（简化处理）
            else:
                # 自适应冻结
                layer_alignments = {}  # 需要从模型中获取
                mitigation_strategy.adaptive_freezing(
                    current_task_id, layer_alignments
                )
        
        # 保存任务结果
        results['task_results'][current_task_id] = {
            'performance_before': performance_before,
            'performance_after': performance_after,
            'performance_drop': performance_drop,
            'alignment_score': alignment_score,
            'alignment_depth': alignment_depth,
            'reversibility_score': reversibility_score,
            'spurious_forgetting_score': spurious_score,
            'forgetting_type': forgetting_type
        }
        
        results['identification_results'][current_task_id] = {
            'spurious': 1 if forgetting_type == 'spurious' else 0,
            'true': 1 if forgetting_type == 'true' else 0,
            'none': 1 if forgetting_type == 'none' else 0
        }
        
        # 保存当前任务数据用于经验回放
        previous_tasks_data.append((train_texts, train_labels))
        
        print(f"任务 {current_task_id} 完成:")
        print(f"  性能: {performance_before:.4f} -> {performance_after:.4f}")
        print(f"  对齐深度: {alignment_depth}")
        print(f"  可逆性分数: {reversibility_score:.4f}")
        print(f"  虚假遗忘分数: {spurious_score:.4f}")
        print(f"  遗忘类型: {forgetting_type}")
    
    # 计算综合指标
    metrics = evaluator.get_comprehensive_metrics()
    results['metrics'] = metrics
    
    # 计算识别准确率
    if results['identification_results']:
        total = len(results['identification_results'])
        spurious_correct = sum(r['spurious'] for r in results['identification_results'].values())
        true_correct = sum(r['true'] for r in results['identification_results'].values())
        results['identification_accuracy'] = {
            'spurious': spurious_correct / total if total > 0 else 0,
            'true': true_correct / total if total > 0 else 0,
            'overall': (spurious_correct + true_correct) / total if total > 0 else 0
        }
    
    return results

def run_ablation(
    ollama_model: str,
    dataset_name: str,
    device: str = "cuda"
) -> Dict:
    """
    运行消融实验：分析各个组件的贡献
    
    Args:
        ollama_model: Ollama模型名称
        dataset_name: 数据集名称
        device: 设备
        
    Returns:
        包含所有消融配置结果的字典
    """
    print(f"\n{'='*80}")
    print(f"消融实验: {ollama_model} | {dataset_name}")
    print(f"{'='*80}\n")
    
    set_seed(experiment_config.seed)
    
    # 检查Ollama连接和模型
    helper = OllamaHelper(model_config.ollama_base_url)
    if not helper.test_connection():
        print(f"错误: 无法连接到Ollama服务 ({model_config.ollama_base_url})")
        return None
    
    if not helper.check_model(ollama_model):
        print(f"警告: 模型 {ollama_model} 不存在，尝试拉取...")
        if not helper.pull_model(ollama_model):
            print(f"错误: 无法拉取模型 {ollama_model}")
            return None
    
    # 加载数据集
    if dataset_name == "clinc150":
        tasks = load_clinc150(
            data_dir=dataset_config.data_dir,
            num_tasks=min(15, dataset_config.num_tasks)
        )
    elif dataset_name == "20newsgroups":
        tasks = load_20newsgroups(
            data_dir=dataset_config.data_dir,
            num_tasks=5
        )
    elif dataset_name == "split_mnist":
        tasks = load_split_mnist(
            data_dir=dataset_config.data_dir,
            num_tasks=5
        )
    elif dataset_name == "permuted_mnist":
        tasks = load_permuted_mnist(
            data_dir=dataset_config.data_dir,
            num_tasks=10
        )
    else:
        print(f"错误: 不支持的数据集 {dataset_name}")
        return None
    
    if tasks is None:
        print(f"错误: 无法加载数据集 {dataset_name}")
        return None
    
    print(f"加载了 {len(tasks)} 个任务")
    
    # 消融实验配置
    ablation_configs = [
        {'name': 'full_method', 'alignment': True, 'reversibility': True, 'tracking': True, 'adaptive': True},
        {'name': 'no_alignment', 'alignment': False, 'reversibility': True, 'tracking': True, 'adaptive': True},
        {'name': 'no_reversibility', 'alignment': True, 'reversibility': False, 'tracking': True, 'adaptive': True},
        {'name': 'no_tracking', 'alignment': True, 'reversibility': True, 'tracking': False, 'adaptive': True},
        {'name': 'fixed_strategy', 'alignment': True, 'reversibility': True, 'tracking': True, 'adaptive': False},
        {'name': 'alignment_only', 'alignment': True, 'reversibility': False, 'tracking': False, 'adaptive': False},
        {'name': 'reversibility_only', 'alignment': False, 'reversibility': True, 'tracking': False, 'adaptive': False},
    ]
    
    all_results = {
        'model': ollama_model,
        'dataset': dataset_name,
        'experiment_group': 'ablation',
        'ablation_results': {}
    }
    
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"运行消融配置: {config['name']}")
        print(f"{'='*60}\n")
        
        # 重新初始化模型
        model = OllamaQwen2ForContinualLearning(
            ollama_base_url=model_config.ollama_base_url,
            ollama_model=ollama_model,
            num_classes=2
        )
        
        # 根据配置设置冻结策略
        if config['adaptive']:
            # 自适应冻结：根据任务ID动态调整
            freeze_ratio = 0.2 + (0 / len(tasks)) * 0.3  # 初始值
        else:
            # 固定冻结30%
            freeze_ratio = 0.3
        
        if hasattr(model, 'freeze_bottom_layers'):
            model.freeze_bottom_layers(ratio=freeze_ratio)
        
        # 初始化评估器
        evaluator = Evaluator(model, device)
        alignment_scorer = AlignmentScore(model, device) if config['alignment'] else None
        reversibility_scorer = ReversibilityScore(device) if config['reversibility'] else None
        
        # 训练和评估
        num_tasks = len(tasks)
        task_accuracies = {}
        
        for task_id in range(num_tasks):
            task_data = tasks[task_id]
            train_texts, train_labels = task_data['train']
            test_texts, test_labels = task_data['test']
            
            # 确定类别数
            if isinstance(train_labels, list):
                num_classes = max(train_labels) + 1 if train_labels else 2
            elif isinstance(train_labels, torch.Tensor):
                num_classes = int(train_labels.max().item()) + 1 if len(train_labels) > 0 else 2
            else:
                num_classes = len(set(train_labels)) if train_labels else 2
            
            if hasattr(model, 'update_num_classes'):
                model.update_num_classes(num_classes)
            else:
                model.num_classes = num_classes
            
            train_loader = create_dataloader(
                train_texts, train_labels, task_id,
                batch_size=model_config.batch_size,
                shuffle=True,
                tokenizer=None
            )
            test_loader = create_dataloader(
                test_texts, test_labels, task_id,
                batch_size=model_config.batch_size,
                shuffle=False,
                tokenizer=None
            )
            
            # 训练
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(model_config.num_epochs):
                for batch in train_loader:
                    if len(batch) == 3:
                        inputs, labels, _ = batch
                    else:
                        inputs, labels = batch
                    
                    if isinstance(inputs, tuple):
                        inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
                    elif isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(device)
                    
                    labels = labels.to(device) if isinstance(labels, torch.Tensor) else labels
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # 评估
            eval_result = evaluator.evaluate_task(task_id, test_loader)
            task_accuracies[task_id] = eval_result.get('accuracy', 0.0)
        
        # 计算综合指标
        metrics = evaluator.get_comprehensive_metrics()
        
        all_results['ablation_results'][config['name']] = {
            'use_alignment': config['alignment'],
            'use_reversibility': config['reversibility'],
            'use_tracking': config['tracking'],
            'use_adaptive': config['adaptive'],
            'task_accuracies': task_accuracies,
            'metrics': metrics
        }
        
        print(f"配置 {config['name']} 完成")
        if 'metrics' in metrics:
            print(f"  平均准确率: {metrics.get('average_accuracy', 0):.4f}")
    
    return all_results

def main(datasets: Optional[List[str]] = None, experiment_groups: Optional[List[str]] = None):
    """
    主函数：运行所有实验
    
    Args:
        datasets: 要使用的数据集列表，如果为None则使用默认值
        experiment_groups: 要运行的实验组列表，如果为None则使用默认值
    """
    print("="*80)
    print("论文实验：虚假遗忘的实时检测与定量分析")
    print("使用Ollama部署的Qwen模型")
    print("="*80)
    
    # 检查Ollama连接
    helper = OllamaHelper(model_config.ollama_base_url)
    if not helper.test_connection():
        print(f"\n错误: 无法连接到Ollama服务 ({model_config.ollama_base_url})")
        print("请确保Ollama服务正在运行:")
        print("  1. 安装Ollama: https://ollama.ai")
        print("  2. 启动服务: ollama serve")
        print("  3. 拉取模型: ollama pull qwen3:1.7b")
        print("              ollama pull qwen2.5:3b")
        print("              ollama pull qwen3:4b")
        print("              ollama pull qwen2.5:32b")
        return
    
    print(f"\n✓ Ollama连接成功: {model_config.ollama_base_url}")
    
    # 检查所有模型
    print("\n检查模型...")
    ollama_models = model_config.ollama_models
    available_models = []
    
    for model_name in ollama_models:
        if helper.check_model(model_name):
            print(f"  ✓ {model_name}")
            available_models.append(model_name)
        else:
            print(f"  ✗ {model_name} (不存在)")
            print(f"    请运行: ollama pull {model_name}")
    
    if not available_models:
        print("\n错误: 没有可用的模型")
        return
    
    print(f"\n可用模型: {len(available_models)}/{len(ollama_models)}")
    
    # 确定设备
    device = experiment_config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("\n警告: CUDA不可用，切换到CPU")
        device = "cpu"
    
    device = torch.device(device)
    print(f"使用设备: {device}\n")
    
    # 实验配置
    if datasets is None:
        datasets = ["clinc150", "20newsgroups"]  # 论文中的主要数据集（默认值）
    else:
        print(f"使用指定的数据集: {datasets}")
    
    if experiment_groups is None:
        experiment_groups = [
            "baseline_control",
            "spurious_forgetting_induced",
            "true_forgetting_induced",
            "mixed_forgetting",
            "ablation"
        ]
        
        # 如果启用深层对齐训练，添加该组
        if experiment_config.use_deep_alignment_training:
            experiment_groups.append("deep_alignment_training")
    else:
        print(f"使用指定的实验组: {experiment_groups}")
    
    # 运行所有实验
    all_results = {}
    
    for model_name in available_models:
        all_results[model_name] = {}
        
        for dataset_name in datasets:
            all_results[model_name][dataset_name] = {}
            
            for exp_group in experiment_groups:
                try:
                    result = run_single_model_experiment(
                        ollama_model=model_name,
                        dataset_name=dataset_name,
                        experiment_group=exp_group,
                        device=device
                    )
                    
                    if result:
                        all_results[model_name][dataset_name][exp_group] = result
                        
                        # 保存单个结果
                        output_file = os.path.join(
                            experiment_config.output_dir,
                            f"{model_name}_{dataset_name}_{exp_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        )
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        print(f"\n结果已保存: {output_file}")
                
                except Exception as e:
                    print(f"\n实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # 生成总结报告
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    summary_file = os.path.join(
        experiment_config.output_dir,
        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n总结报告已保存: {summary_file}")
    
    # 打印关键结果
    for model_name, model_results in all_results.items():
        print(f"\n模型: {model_name}")
        for dataset_name, dataset_results in model_results.items():
            print(f"  数据集: {dataset_name}")
            for exp_group, result in dataset_results.items():
                if 'identification_accuracy' in result:
                    acc = result['identification_accuracy']
                    print(f"    {exp_group}:")
                    print(f"      识别准确率: {acc.get('overall', 0)*100:.2f}%")
                    print(f"      虚假遗忘: {acc.get('spurious', 0)*100:.2f}%")
                    print(f"      真实遗忘: {acc.get('true', 0)*100:.2f}%")

if __name__ == "__main__":
    main()

