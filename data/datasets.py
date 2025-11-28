"""
数据集加载模块
支持多种持续学习数据集
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

# 自动配置 HuggingFace 镜像（如果未设置）
# 这样可以避免网络连接问题，特别是在中国大陆地区
# 注意：这个设置只在当前 Python 进程有效
if 'HF_ENDPOINT' not in os.environ:
    # 使用 HuggingFace 中国镜像站点
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 不在这里打印提示，避免每次导入都显示，改为在需要时显示

from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer

class ContinualLearningDataset(Dataset):
    """持续学习数据集基类"""
    def __init__(self, data: List, labels: List, task_id: int):
        self.data = data
        self.labels = labels
        self.task_id = task_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.task_id

def load_clinc150(data_dir: str = "./data", num_tasks: int = 15) -> Dict:
    """
    加载CLINC-150数据集
    150个意图分类任务，可以分成多个持续学习任务
    """
    print("Loading CLINC-150 dataset...")
    
    # 尝试从本地缓存加载
    cache_dir = os.path.join(data_dir, "clinc150_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 首先尝试从本地缓存加载
    try:
        print("检查本地缓存...")
        from datasets import load_from_disk
        dataset = load_from_disk(cache_dir)
        print("✓ 从本地缓存加载成功")
    except:
        # 本地缓存不存在，尝试从 HuggingFace 下载
        print("本地缓存不存在，尝试从 HuggingFace 下载...")
        
        # 显示当前使用的端点（让用户知道已自动配置镜像）
        hf_endpoint = os.environ.get('HF_ENDPOINT')
        if hf_endpoint:
            if 'mirror' in hf_endpoint.lower() or 'hf-mirror' in hf_endpoint:
                print(f"✓ 已自动配置使用 HuggingFace 镜像: {hf_endpoint}")
            else:
                print(f"使用 HuggingFace 端点: {hf_endpoint}")
        else:
            print("使用 HuggingFace 官方源（可能需要代理）")
        
        try:
            # 直接加载数据集（不使用 DownloadConfig，因为参数可能不兼容）
            dataset = load_dataset(
                "clinc_oos", 
                "plus",
                cache_dir=cache_dir
            )
            print("✓ 数据集下载并加载成功")
            
            # 保存到本地缓存以便下次使用
            try:
                dataset.save_to_disk(cache_dir)
                print(f"✓ 数据集已保存到本地缓存: {cache_dir}")
            except:
                pass
                
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ 下载失败: {error_msg[:200]}...")
            print("\n可能的解决方案：")
            print("1. 检查网络连接")
            print(f"2. 当前使用的端点: {os.environ.get('HF_ENDPOINT', '默认')}")
            print("   如果需要更换镜像，可以在导入数据集模块前设置:")
            print("   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'")
            print("   或使用其他镜像站点")
            print(f"3. 检查缓存目录: {cache_dir}")
            print("4. 使用其他数据集（如 20newsgroups）进行实验")
            print("5. 或稍后重试，可能是临时网络问题")
            return None
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    # CLINC-150 数据集结构：只有 text 和 intent 字段
    # intent 是 ClassLabel 类型，但访问时返回的是整数索引（0-149）
    # 按意图分组，然后分成多个任务
    
    # 收集所有唯一的意图
    all_intents = set()
    intent_to_samples = {}
    
    # 处理训练数据
    for item in train_data:
        intent_val = item['intent']
        # 确保转换为整数（ClassLabel 类型在访问时返回整数索引）
        intent_idx = int(intent_val) if not isinstance(intent_val, int) else intent_val
        
        if intent_idx not in intent_to_samples:
            intent_to_samples[intent_idx] = {'train': [], 'test': []}
            all_intents.add(intent_idx)
        
        intent_to_samples[intent_idx]['train'].append(item['text'])
    
    # 处理测试数据
    for item in test_data:
        intent_val = item['intent']
        intent_idx = int(intent_val) if not isinstance(intent_val, int) else intent_val
        
        if intent_idx not in intent_to_samples:
            intent_to_samples[intent_idx] = {'train': [], 'test': []}
        
        intent_to_samples[intent_idx]['test'].append(item['text'])
    
    # 将意图分成 num_tasks 个任务组
    intent_list = sorted(list(all_intents))
    intents_per_task = len(intent_list) // num_tasks
    if intents_per_task == 0:
        intents_per_task = 1
    
    tasks = {}
    
    for task_id in range(num_tasks):
        start_idx = task_id * intents_per_task
        end_idx = (task_id + 1) * intents_per_task if task_id < num_tasks - 1 else len(intent_list)
        task_intents = intent_list[start_idx:end_idx]
        
        # 收集该任务的所有文本和标签
        task_texts_train = []
        task_labels_train = []
        task_texts_test = []
        task_labels_test = []
        
        # 每个意图在任务中的本地标签ID（从0开始）
        intent_to_local_id = {intent: idx for idx, intent in enumerate(task_intents)}
        
        for intent_idx in task_intents:
            local_label = intent_to_local_id[intent_idx]
            
            # 添加训练数据
            for text in intent_to_samples[intent_idx]['train']:
                task_texts_train.append(text)
                task_labels_train.append(local_label)
            
            # 添加测试数据
            for text in intent_to_samples[intent_idx]['test']:
                task_texts_test.append(text)
                task_labels_test.append(local_label)
        
        tasks[task_id] = {
            'train': (task_texts_train, task_labels_train),
            'test': (task_texts_test, task_labels_test)
        }
        
        print(f"任务 {task_id}: {len(task_intents)} 个意图, {len(task_texts_train)} 个训练样本, {len(task_texts_test)} 个测试样本")
    
    return tasks

def load_20newsgroups(data_dir: str = "./data", num_tasks: int = 5) -> Dict:
    """
    加载20 Newsgroups数据集
    分成多个任务
    """
    print("Loading 20 Newsgroups dataset...")
    
    # 获取所有类别
    categories = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
        'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        'sci.space', 'soc.religion.christian', 'talk.politics.guns',
        'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
    ]
    
    # 加载数据
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42
    )
    
    # 将20个类别分成num_tasks个任务
    categories_per_task = len(categories) // num_tasks
    tasks = {}
    
    for task_id in range(num_tasks):
        start_idx = task_id * categories_per_task
        end_idx = (task_id + 1) * categories_per_task if task_id < num_tasks - 1 else len(categories)
        task_categories = categories[start_idx:end_idx]
        
        # 筛选属于当前任务的样本
        train_mask = [newsgroups_train.target_names[cat] in task_categories for cat in newsgroups_train.target]
        test_mask = [newsgroups_test.target_names[cat] in task_categories for cat in newsgroups_test.target]
        
        train_texts = [newsgroups_train.data[i] for i in range(len(newsgroups_train.data)) if train_mask[i]]
        train_labels = [newsgroups_train.target[i] for i in range(len(newsgroups_train.target)) if train_mask[i]]
        
        test_texts = [newsgroups_test.data[i] for i in range(len(newsgroups_test.data)) if test_mask[i]]
        test_labels = [newsgroups_test.target[i] for i in range(len(newsgroups_test.target)) if test_mask[i]]
        
        # 重新映射标签为0开始
        unique_labels = sorted(list(set(train_labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        train_labels = [label_to_id[label] for label in train_labels]
        test_labels = [label_to_id[label] for label in test_labels if label in label_to_id]
        
        tasks[task_id] = {
            'train': (train_texts, train_labels),
            'test': (test_texts, test_labels)
        }
    
    return tasks

def load_split_mnist(data_dir: str = "./data", num_tasks: int = 5) -> Dict:
    """
    加载Split-MNIST数据集
    将MNIST的10个类别分成5个任务，每个任务2个类别
    """
    print("Loading Split-MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载完整MNIST
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    tasks = {}
    classes_per_task = 10 // num_tasks
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        
        # 筛选属于当前任务的样本
        train_mask = [(train_dataset.targets[i] >= start_class) and 
                      (train_dataset.targets[i] < end_class) 
                      for i in range(len(train_dataset))]
        test_mask = [(test_dataset.targets[i] >= start_class) and 
                     (test_dataset.targets[i] < end_class) 
                     for i in range(len(test_dataset))]
        
        train_indices = [i for i in range(len(train_dataset)) if train_mask[i]]
        test_indices = [i for i in range(len(test_dataset)) if test_mask[i]]
        
        # 创建子数据集
        train_data = [train_dataset[i][0] for i in train_indices]
        train_labels = [train_dataset[i][1] - start_class for i in train_indices]  # 重新映射标签
        
        test_data = [test_dataset[i][0] for i in test_indices]
        test_labels = [test_dataset[i][1] - start_class for i in test_indices]
        
        tasks[task_id] = {
            'train': (train_data, train_labels),
            'test': (test_data, test_labels)
        }
    
    return tasks

def load_permuted_mnist(data_dir: str = "./data", num_tasks: int = 10) -> Dict:
    """
    加载Permuted-MNIST数据集
    通过像素重排创建多个任务
    """
    print("Loading Permuted-MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    base_train = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=None
    )
    base_test = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=None
    )
    
    tasks = {}
    
    for task_id in range(num_tasks):
        # 为每个任务生成随机排列
        np.random.seed(task_id)
        permutation = np.random.permutation(28 * 28)
        
        def permute_image(image):
            image = np.array(image).flatten()
            image = image[permutation]
            image = image.reshape(28, 28)
            return transform(image)
        
        train_data = [permute_image(base_train[i][0]) for i in range(len(base_train))]
        train_labels = [base_train[i][1] for i in range(len(base_train))]
        
        test_data = [permute_image(base_test[i][0]) for i in range(len(base_test))]
        test_labels = [base_test[i][1] for i in range(len(base_test))]
        
        tasks[task_id] = {
            'train': (train_data, train_labels),
            'test': (test_data, test_labels)
        }
    
    return tasks

def load_split_cifar10(data_dir: str = "./data", num_tasks: int = 5) -> Dict:
    """
    加载Split-CIFAR-10数据集
    将CIFAR-10的10个类别分成5个任务
    """
    print("Loading Split-CIFAR-10 dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    tasks = {}
    classes_per_task = 10 // num_tasks
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task
        
        train_mask = [(train_dataset.targets[i] >= start_class) and 
                      (train_dataset.targets[i] < end_class) 
                      for i in range(len(train_dataset))]
        test_mask = [(test_dataset.targets[i] >= start_class) and 
                     (test_dataset.targets[i] < end_class) 
                     for i in range(len(test_dataset))]
        
        train_indices = [i for i in range(len(train_dataset)) if train_mask[i]]
        test_indices = [i for i in range(len(test_dataset)) if test_mask[i]]
        
        train_data = [train_dataset[i][0] for i in train_indices]
        train_labels = [train_dataset[i][1] - start_class for i in train_indices]
        
        test_data = [test_dataset[i][0] for i in test_indices]
        test_labels = [test_dataset[i][1] - start_class for i in test_indices]
        
        tasks[task_id] = {
            'train': (train_data, train_labels),
            'test': (test_data, test_labels)
        }
    
    return tasks

def create_dataloader(data, labels, task_id, batch_size=32, shuffle=True, tokenizer=None):
    """
    创建DataLoader
    """
    if tokenizer is not None:
        # 文本数据需要tokenize
        encoded = tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        dataset = ContinualLearningDataset(
            list(zip(encoded['input_ids'], encoded['attention_mask'])),
            labels,
            task_id
        )
    else:
        # 图像数据
        dataset = ContinualLearningDataset(data, labels, task_id)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

