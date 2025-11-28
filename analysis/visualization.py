"""
可视化工具
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict

def visualize_representations(
    representations: List[torch.Tensor],
    labels: List[int],
    method: str = "pca",
    save_path: str = None
):
    """
    可视化表示空间
    
    Args:
        representations: 表示列表
        labels: 标签列表
        method: 降维方法 ('pca' or 'tsne')
        save_path: 保存路径
    """
    # 合并所有表示
    all_reps = torch.cat(representations, dim=0).cpu().numpy()
    all_labels = np.array(labels)
    
    # 降维
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(all_reps)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'Representation Space Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_alignment_heatmap(
    alignment_scores: Dict[str, List[float]],
    save_path: str = None
):
    """
    绘制对齐度热力图
    
    Args:
        alignment_scores: {layer_name: [scores]}
        save_path: 保存路径
    """
    import seaborn as sns
    
    # 准备数据
    layer_names = list(alignment_scores.keys())
    max_length = max(len(scores) for scores in alignment_scores.values())
    
    heatmap_data = np.zeros((len(layer_names), max_length))
    
    for i, layer_name in enumerate(layer_names):
        scores = alignment_scores[layer_name]
        heatmap_data[i, :len(scores)] = scores
    
    # 绘制
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=range(max_length),
        yticklabels=layer_names,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        annot=False
    )
    plt.title('Alignment Heatmap')
    plt.xlabel('Time Step')
    plt.ylabel('Layer')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

