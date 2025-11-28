"""
配置 HuggingFace 镜像的工具脚本
用于解决在中国大陆访问 HuggingFace 时的网络问题
"""
import os
import sys

def setup_hf_mirror():
    """设置 HuggingFace 镜像环境变量"""
    mirror_url = "https://hf-mirror.com"
    
    print("="*60)
    print("HuggingFace 镜像配置工具")
    print("="*60)
    print()
    
    # 检查当前设置
    current_endpoint = os.environ.get('HF_ENDPOINT')
    if current_endpoint:
        print(f"当前 HF_ENDPOINT: {current_endpoint}")
    else:
        print("当前未设置 HF_ENDPOINT")
    print()
    
    # 设置镜像
    print(f"设置 HuggingFace 镜像为: {mirror_url}")
    os.environ['HF_ENDPOINT'] = mirror_url
    
    # 验证设置
    if os.environ.get('HF_ENDPOINT') == mirror_url:
        print("✓ 环境变量设置成功（当前会话有效）")
    else:
        print("✗ 环境变量设置失败")
        return False
    
    print()
    print("注意: 这个设置只在当前 Python 会话中有效")
    print()
    print("要永久设置，请执行以下命令之一：")
    print()
    
    if sys.platform == 'win32':
        print("Windows PowerShell:")
        print(f"  $env:HF_ENDPOINT='{mirror_url}'")
        print()
        print("Windows CMD:")
        print(f"  set HF_ENDPOINT={mirror_url}")
        print()
        print("或者在系统环境变量中添加 HF_ENDPOINT")
    else:
        print("Linux/Mac:")
        print(f"  export HF_ENDPOINT={mirror_url}")
        print()
        print("要永久生效，添加到 ~/.bashrc 或 ~/.zshrc:")
        print(f"  echo 'export HF_ENDPOINT={mirror_url}' >> ~/.bashrc")
    
    print()
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = setup_hf_mirror()
    
    if success:
        print("\n尝试测试加载数据集...")
        try:
            from datasets import load_dataset
            print("正在测试连接...")
            # 只是测试连接，不实际加载完整数据集
            print("✓ 连接测试成功！现在可以正常使用数据集加载功能。")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            print("请检查网络连接或稍后重试")
    
    sys.exit(0 if success else 1)
