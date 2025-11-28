"""
运行实验的主脚本
对应论文中的实验设置，使用Ollama部署的4个Qwen模型

论文对应：
- Section 5.1: Experimental Setup
- Section 5.2: Experimental Results
- Table 1-4: 实验结果表格
"""
import argparse
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import model_config, dataset_config, experiment_config
from experiments.main_experiment import main

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行论文实验：虚假遗忘的实时检测与定量分析（对应论文Section 5）'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama服务地址（默认: http://localhost:11434）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='设备（默认: cuda）'
    )
    
    parser.add_argument(
        '--use-deep-alignment',
        action='store_true',
        default=True,
        help='使用深层对齐训练（论文默认启用，对应Section 4.3）'
    )
    
    parser.add_argument(
        '--no-deep-alignment',
        dest='use_deep_alignment',
        action='store_false',
        help='禁用深层对齐训练'
    )
    
    parser.add_argument(
        '--use-hybrid-strategy',
        action='store_true',
        default=True,
        help='使用混合缓解策略（论文默认启用，对应Algorithm 2）'
    )
    
    parser.add_argument(
        '--no-hybrid-strategy',
        dest='use_hybrid_strategy',
        action='store_false',
        help='禁用混合缓解策略'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['clinc150', '20newsgroups'],
        choices=['clinc150', '20newsgroups', 'split_mnist', 'permuted_mnist'],
        help='要使用的数据集（默认: clinc150 20newsgroups，对应论文Table 1-2）'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='要使用的模型列表（默认: 使用所有4个模型：qwen3:1.7b qwen2.5:3b qwen3:4b qwen2.5:32b）'
    )
    
    parser.add_argument(
        '--experiment-groups',
        type=str,
        nargs='+',
        default=None,
        help='要运行的实验组（默认: 所有组）'
    )
    
    parser.add_argument(
        '--check-ollama',
        action='store_true',
        help='仅检查Ollama连接和模型，不运行实验'
    )
    
    return parser.parse_args()

def check_ollama_setup():
    """检查Ollama设置"""
    from utils.ollama_helper import OllamaHelper
    
    print("="*80)
    print("检查Ollama设置")
    print("="*80)
    
    helper = OllamaHelper(model_config.ollama_base_url)
    
    # 检查连接
    if not helper.test_connection():
        print(f"\n✗ 无法连接到Ollama服务 ({model_config.ollama_base_url})")
        print("\n请确保Ollama服务正在运行：")
        print("  1. 安装Ollama: https://ollama.ai")
        print("  2. 启动服务: ollama serve")
        return False
    
    print(f"\n✓ Ollama连接成功: {model_config.ollama_base_url}")
    
    # 检查模型
    print("\n检查论文中的4个模型：")
    ollama_models = model_config.ollama_models
    available_models = []
    missing_models = []
    
    for model_name in ollama_models:
        if helper.check_model(model_name):
            print(f"  ✓ {model_name}")
            available_models.append(model_name)
        else:
            print(f"  ✗ {model_name} (不存在)")
            missing_models.append(model_name)
    
    if missing_models:
        print(f"\n缺少以下模型，请运行以下命令拉取：")
        for model_name in missing_models:
            print(f"  ollama pull {model_name}")
        return False
    
    print(f"\n✓ 所有模型可用: {len(available_models)}/{len(ollama_models)}")
    return True

def main_with_args():
    """带参数的主函数"""
    args = parse_args()
    
    # 更新配置
    model_config.ollama_base_url = args.ollama_url
    model_config.use_ollama = True  # 论文实验必须使用Ollama
    experiment_config.device = args.device
    experiment_config.use_deep_alignment_training = args.use_deep_alignment
    experiment_config.use_hybrid_strategy = args.use_hybrid_strategy
    
    # 如果指定了模型，更新模型列表
    if args.models:
        model_config.ollama_models = args.models
        print(f"使用指定模型: {args.models}")
    
    # 如果仅检查Ollama
    if args.check_ollama:
        if check_ollama_setup():
            print("\n✓ Ollama设置检查通过，可以运行实验")
        else:
            print("\n✗ Ollama设置检查失败，请先解决上述问题")
        return
    
    # 检查Ollama设置
    if not check_ollama_setup():
        print("\n✗ Ollama设置检查失败，请先解决上述问题")
        print("\n提示：可以使用 --check-ollama 仅检查Ollama设置")
        return
    
    # 打印配置
    print("\n" + "="*80)
    print("论文实验配置")
    print("="*80)
    print(f"Ollama服务: {model_config.ollama_base_url}")
    print(f"设备: {experiment_config.device}")
    print(f"使用深层对齐训练: {experiment_config.use_deep_alignment_training} (对应论文Section 4.3)")
    print(f"使用混合策略: {experiment_config.use_hybrid_strategy} (对应论文Algorithm 2)")
    print(f"模型列表: {model_config.ollama_models} (对应论文Section 5.1)")
    print(f"数据集: {args.datasets} (对应论文Table 1-2)")
    print(f"对齐阈值: τ_align={experiment_config.alignment_threshold}, "
          f"τ_R={experiment_config.reversibility_threshold}, "
          f"τ_S={experiment_config.spurious_forgetting_threshold} (对应论文Section 3.4)")
    print("="*80)
    print()
    
    # 运行实验
    print("开始运行论文实验...")
    print("实验对应论文：")
    print("  - Section 5.1: Experimental Setup")
    print("  - Section 5.2: Experimental Results")
    print("  - Table 1-4: 实验结果表格")
    print()
    
    # 传递数据集和实验组参数
    # argparse 会将 --experiment-groups 转换为 experiment_groups 属性
    experiment_groups = getattr(args, 'experiment_groups', None)
    main(
        datasets=args.datasets,
        experiment_groups=experiment_groups if experiment_groups else None
    )

if __name__ == "__main__":
    main_with_args()

