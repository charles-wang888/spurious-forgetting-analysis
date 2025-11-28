"""
CKA (Centered Kernel Alignment) 完整实现
基于论文: "Similarity of Neural Network Representations Revisited"
开源实现参考: https://github.com/google-research/google-research/tree/master/representation_similarity
"""
import torch
import numpy as np
from typing import Union, Optional
from scipy.linalg import eigh


def linear_cka(X: Union[torch.Tensor, np.ndarray], 
                Y: Union[torch.Tensor, np.ndarray],
                debiased: bool = False) -> float:
    """
    计算线性CKA (Linear Centered Kernel Alignment)
    
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    
    Args:
        X: 表示矩阵1 [n_samples, n_features]
        Y: 表示矩阵2 [n_samples, n_features]
        debiased: 是否使用去偏版本
    
    Returns:
        CKA分数 [0, 1]
    """
    # 转换为numpy（如果需要）
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    # 中心化
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    if debiased:
        # 去偏版本（适用于小样本）
        n = X.shape[0]
        # 计算去偏的HSIC
        hsic = np.trace(X @ Y.T @ Y @ X.T) - (1.0 / (n - 2)) * np.trace(X @ Y.T) * np.trace(Y @ X.T)
        hsic = hsic / ((n - 1) * (n - 2))
        
        norm1 = np.trace(X @ X.T @ X @ X.T) - (1.0 / (n - 2)) * (np.trace(X @ X.T) ** 2)
        norm1 = norm1 / ((n - 1) * (n - 2))
        
        norm2 = np.trace(Y @ Y.T @ Y @ Y.T) - (1.0 / (n - 2)) * (np.trace(Y @ Y.T) ** 2)
        norm2 = norm2 / ((n - 1) * (n - 2))
    else:
        # 标准版本
        hsic = np.trace(X @ Y.T @ Y @ X.T)
        norm1 = np.trace(X @ X.T @ X @ X.T)
        norm2 = np.trace(Y @ Y.T @ Y @ Y.T)
    
    if norm1 > 1e-10 and norm2 > 1e-10:
        cka = hsic / np.sqrt(norm1 * norm2)
        return float(cka)
    else:
        return 0.0


def rbf_cka(X: Union[torch.Tensor, np.ndarray],
            Y: Union[torch.Tensor, np.ndarray],
            sigma: Optional[float] = None,
            debiased: bool = False) -> float:
    """
    计算RBF CKA (使用RBF核的CKA)
    
    Args:
        X: 表示矩阵1 [n_samples, n_features]
        Y: 表示矩阵2 [n_samples, n_features]
        sigma: RBF核的带宽参数（如果为None，则自动选择）
        debiased: 是否使用去偏版本
    
    Returns:
        CKA分数 [0, 1]
    """
    # 转换为numpy
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    # 中心化
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # 自动选择sigma（使用中位数启发式）
    if sigma is None:
        # 计算X和Y的成对距离
        from scipy.spatial.distance import pdist, squareform
        dists_X = pdist(X)
        dists_Y = pdist(Y)
        sigma = np.median(np.concatenate([dists_X, dists_Y]))
    
    # 计算RBF核矩阵
    from scipy.spatial.distance import cdist
    K_X = np.exp(-cdist(X, X) ** 2 / (2 * sigma ** 2))
    K_Y = np.exp(-cdist(Y, Y) ** 2 / (2 * sigma ** 2))
    
    # 中心化核矩阵
    n = K_X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_X = H @ K_X @ H
    K_Y = H @ K_Y @ H
    
    if debiased:
        # 去偏版本
        hsic = np.trace(K_X @ K_Y) - (1.0 / (n - 2)) * np.trace(K_X) * np.trace(K_Y)
        hsic = hsic / ((n - 1) * (n - 2))
        
        norm1 = np.trace(K_X @ K_X) - (1.0 / (n - 2)) * (np.trace(K_X) ** 2)
        norm1 = norm1 / ((n - 1) * (n - 2))
        
        norm2 = np.trace(K_Y @ K_Y) - (1.0 / (n - 2)) * (np.trace(K_Y) ** 2)
        norm2 = norm2 / ((n - 1) * (n - 2))
    else:
        # 标准版本
        hsic = np.trace(K_X @ K_Y)
        norm1 = np.trace(K_X @ K_X)
        norm2 = np.trace(K_Y @ K_Y)
    
    if norm1 > 1e-10 and norm2 > 1e-10:
        cka = hsic / np.sqrt(norm1 * norm2)
        return float(cka)
    else:
        return 0.0


def compute_cka_matrix(representations: list,
                       kernel: str = "linear",
                       debiased: bool = False) -> np.ndarray:
    """
    计算多个表示之间的CKA矩阵
    
    Args:
        representations: 表示列表，每个元素是 [n_samples, n_features]
        kernel: 核类型 ('linear' or 'rbf')
        debiased: 是否使用去偏版本
    
    Returns:
        CKA矩阵 [n_representations, n_representations]
    """
    n = len(representations)
    cka_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if kernel == "linear":
                cka = linear_cka(representations[i], representations[j], debiased=debiased)
            elif kernel == "rbf":
                cka = rbf_cka(representations[i], representations[j], debiased=debiased)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka  # 对称矩阵
    
    return cka_matrix


class CKA:
    """
    CKA计算器类（封装版本）
    """
    
    def __init__(self, kernel: str = "linear", debiased: bool = False):
        """
        初始化CKA计算器
        
        Args:
            kernel: 核类型 ('linear' or 'rbf')
            debiased: 是否使用去偏版本
        """
        self.kernel = kernel
        self.debiased = debiased
    
    def compute(self, X: Union[torch.Tensor, np.ndarray],
                Y: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算两个表示之间的CKA
        
        Args:
            X: 表示1 [n_samples, n_features]
            Y: 表示2 [n_samples, n_features]
        
        Returns:
            CKA分数 [0, 1]
        """
        if self.kernel == "linear":
            return linear_cka(X, Y, debiased=self.debiased)
        elif self.kernel == "rbf":
            return rbf_cka(X, Y, debiased=self.debiased)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def compute_matrix(self, representations: list) -> np.ndarray:
        """
        计算多个表示之间的CKA矩阵
        
        Args:
            representations: 表示列表
        
        Returns:
            CKA矩阵
        """
        return compute_cka_matrix(representations, kernel=self.kernel, debiased=self.debiased)


# 使用示例
if __name__ == "__main__":
    # 示例：计算两个表示之间的CKA
    X = np.random.randn(100, 512)  # 100个样本，512维特征
    Y = np.random.randn(100, 512)
    
    cka_calculator = CKA(kernel="linear", debiased=False)
    cka_score = cka_calculator.compute(X, Y)
    print(f"Linear CKA: {cka_score:.4f}")
    
    # 示例：计算多个表示之间的CKA矩阵
    representations = [np.random.randn(100, 512) for _ in range(5)]
    cka_matrix = cka_calculator.compute_matrix(representations)
    print(f"\nCKA Matrix shape: {cka_matrix.shape}")
    print(f"CKA Matrix:\n{cka_matrix}")

