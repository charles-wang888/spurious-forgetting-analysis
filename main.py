"""
主程序入口
"""
import argparse
import sys
import os
import torch

def main():
    parser = argparse.ArgumentParser(description='虚假遗忘识别与缓解机制实验')
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['exp1', 'exp2', 'exp3', 'exp4'],
        default='exp1',
        help='选择要运行的实验'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['clinc150', '20newsgroups', 'split_mnist', 'permuted_mnist', 'split_cifar10'],
        default='20newsgroups',
        help='选择数据集'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['qwen2.5', 'ollama_qwen2.5'],
        default='ollama_qwen2.5',
        help='选择模型 (qwen2.5, ollama_qwen2.5) - 论文实验使用Ollama Qwen模型'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda or cpu)'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama服务地址（仅当使用ollama_qwen2.5时）'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default='qwen2.5:32b',
        help='Ollama模型名称（仅当使用ollama_qwen2.5时）'
    )
    
    args = parser.parse_args()
    
    # 更新配置
    from config import model_config
    model_config.model_type = args.model
    if args.model == 'ollama_qwen2.5':
        model_config.ollama_base_url = args.ollama_url
        model_config.ollama_model = args.ollama_model
        model_config.use_ollama = True
        print(f"使用Ollama模型: {args.ollama_model}")
        print(f"Ollama服务地址: {args.ollama_url}")
    
    # 根据实验类型运行相应的脚本
    if args.experiment == 'exp1':
        from experiments.spurious_forgetting_identification import main as exp1_main
        # 检查CUDA是否可用，如果不可用则使用CPU
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                print("\n" + "="*60)
                print("警告: CUDA不可用，切换到CPU模式")
                print("="*60)
                print("\n可能的原因:")
                print("  1. PyTorch未安装CUDA版本")
                print("  2. CUDA驱动未正确安装")
                print("  3. GPU硬件问题")
                print("\n检查方法:")
                print("  - 运行: python -c \"import torch; print(torch.cuda.is_available())\"")
                print("  - 检查: nvidia-smi (如果安装了NVIDIA驱动)")
                print("\n解决方案:")
                print("  - 安装CUDA版本的PyTorch:")
                print("    https://pytorch.org/get-started/locally/")
                print("  - 或使用CPU模式: --device cpu")
                print("="*60 + "\n")
                args.device = 'cpu'
            else:
                print(f"\n✓ CUDA可用")
                print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA版本: {torch.version.cuda}")
                print(f"  PyTorch版本: {torch.__version__}\n")
        exp1_main(device=args.device)
    elif args.experiment == 'exp2':
        print("实验2：机制探索（待实现）")
    elif args.experiment == 'exp3':
        print("实验3：缓解策略效果验证（待实现）")
    elif args.experiment == 'exp4':
        print("="*60)
        print("实验4：消融实验")
        print("="*60)
        print("\n⚠️  注意：消融实验已整合到 run_experiments.py")
        print("请使用以下命令运行消融实验：")
        print("  python run_experiments.py --experiment-groups ablation")
        print("\n或者使用 main.py 的旧接口（已弃用）：")
        print("="*60)
        # 为了向后兼容，仍然可以运行，但建议使用新接口
        from experiments.main_experiment import run_ablation
        from config import dataset_config, experiment_config
        
        # 使用默认模型和数据集
        model_name = args.ollama_model if args.model == 'ollama_qwen2.5' else 'qwen2.5:3b'
        dataset_name = args.dataset
        
        result = run_ablation(
            ollama_model=model_name,
            dataset_name=dataset_name,
            device=args.device
        )
        
        if result:
            print("\n消融实验完成！")
            # 保存结果
            import json
            from datetime import datetime
            output_file = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果已保存: {output_file}")

if __name__ == "__main__":
    main()

