"""
遗忘指标计算模块
计算持续学习中的各种遗忘指标
"""
import numpy as np
from typing import List, Dict, Tuple

def compute_forgetting_measure(
    accuracies: Dict[int, List[float]]
) -> Dict[int, float]:
    """
    计算遗忘度量（Forgetting Measure）
    
    FM_i = max_{k in {1, ..., t-1}} a_{k,i} - a_{t,i}
    
    Args:
        accuracies: {task_id: [accuracy_after_task_1, accuracy_after_task_2, ...]}
    
    Returns:
        {task_id: forgetting_measure}
    """
    forgetting_measures = {}
    
    for task_id, acc_list in accuracies.items():
        if len(acc_list) > 1:
            # 找到该任务在历史中的最高准确率
            max_acc = max(acc_list[:-1]) if len(acc_list) > 1 else acc_list[0]
            # 当前准确率
            current_acc = acc_list[-1]
            # 遗忘度量
            forgetting_measures[task_id] = max_acc - current_acc
        else:
            forgetting_measures[task_id] = 0.0
    
    return forgetting_measures

def compute_backward_transfer(
    accuracies: Dict[int, List[float]]
) -> float:
    """
    计算后向迁移（Backward Transfer）
    
    BWT = (1/(T-1)) * sum_{i=1}^{T-1} (a_{T,i} - a_{i,i})
    
    Args:
        accuracies: {task_id: [accuracy_after_task_1, accuracy_after_task_2, ...]}
    
    Returns:
        后向迁移分数
    """
    if len(accuracies) < 2:
        return 0.0
    
    bwt_scores = []
    T = len(accuracies)
    
    for i in range(T - 1):
        task_id = i
        if task_id in accuracies and len(accuracies[task_id]) > i:
            # 任务i在训练任务i时的准确率
            acc_after_i = accuracies[task_id][i] if len(accuracies[task_id]) > i else 0.0
            # 任务i在训练完所有任务后的准确率
            acc_after_T = accuracies[task_id][-1] if len(accuracies[task_id]) > T - 1 else acc_after_i
            
            bwt = acc_after_T - acc_after_i
            bwt_scores.append(bwt)
    
    if len(bwt_scores) > 0:
        return np.mean(bwt_scores)
    else:
        return 0.0

def compute_forward_transfer(
    accuracies: Dict[int, List[float]]
) -> float:
    """
    计算前向迁移（Forward Transfer）
    
    FWT = (1/T) * sum_{i=1}^T (a_{i,i} - a_{random,i})
    
    其中a_{random,i}是随机初始化的模型在任务i上的准确率
    
    Args:
        accuracies: {task_id: [accuracy_after_task_1, accuracy_after_task_2, ...]}
    
    Returns:
        前向迁移分数（简化版，假设随机准确率为0）
    """
    if len(accuracies) == 0:
        return 0.0
    
    fwt_scores = []
    
    for task_id, acc_list in accuracies.items():
        if len(acc_list) > task_id:
            # 任务task_id在训练任务task_id时的准确率
            acc_after_task = acc_list[task_id]
            # 假设随机准确率为0（对于分类任务，应该是1/num_classes）
            fwt_scores.append(acc_after_task)
    
    if len(fwt_scores) > 0:
        return np.mean(fwt_scores)
    else:
        return 0.0

def compute_average_accuracy(
    accuracies: Dict[int, List[float]]
) -> float:
    """
    计算平均准确率（Average Accuracy）
    
    ACC = (1/T) * sum_{i=1}^T a_{T,i}
    
    Args:
        accuracies: {task_id: [accuracy_after_task_1, accuracy_after_task_2, ...]}
    
    Returns:
        平均准确率
    """
    if len(accuracies) == 0:
        return 0.0
    
    final_accuracies = []
    
    for task_id, acc_list in accuracies.items():
        if len(acc_list) > 0:
            final_accuracies.append(acc_list[-1])
    
    if len(final_accuracies) > 0:
        return np.mean(final_accuracies)
    else:
        return 0.0

def compute_forgetting_rate(
    accuracies: Dict[int, List[float]]
) -> Dict[int, float]:
    """
    计算遗忘率（Forgetting Rate）
    
    FR_i = (max_acc_i - current_acc_i) / max_acc_i
    
    Args:
        accuracies: {task_id: [accuracy_after_task_1, accuracy_after_task_2, ...]}
    
    Returns:
        {task_id: forgetting_rate}
    """
    forgetting_rates = {}
    
    for task_id, acc_list in accuracies.items():
        if len(acc_list) > 1:
            max_acc = max(acc_list[:-1])
            current_acc = acc_list[-1]
            if max_acc > 1e-6:
                forgetting_rates[task_id] = (max_acc - current_acc) / max_acc
            else:
                forgetting_rates[task_id] = 0.0
        else:
            forgetting_rates[task_id] = 0.0
    
    return forgetting_rates

