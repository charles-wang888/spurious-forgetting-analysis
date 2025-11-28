"""
Ollama辅助工具
用于测试和管理Ollama服务
"""
import requests
import json
from typing import List, Dict, Optional

class OllamaHelper:
    """Ollama服务辅助类"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except:
            return []
    
    def check_model(self, model_name: str) -> bool:
        """检查模型是否可用"""
        models = self.list_models()
        return model_name in models
    
    def pull_model(self, model_name: str) -> bool:
        """拉取模型（如果不存在）"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            if response.status_code == 200:
                # 读取流式响应
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("status") == "success":
                            return True
            return False
        except Exception as e:
            print(f"拉取模型失败: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """获取模型信息"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

def setup_ollama_model(model_name: str = "qwen2.5:32b", base_url: str = "http://localhost:11434"):
    """
    设置Ollama模型
    
    Args:
        model_name: 模型名称
        base_url: Ollama服务地址
    
    Returns:
        是否设置成功
    """
    helper = OllamaHelper(base_url)
    
    print(f"检查Ollama连接: {base_url}")
    if not helper.test_connection():
        print("✗ 无法连接到Ollama服务")
        print("请确保Ollama服务正在运行:")
        print("  1. 安装Ollama: https://ollama.ai")
        print("  2. 启动服务: ollama serve")
        return False
    
    print("✓ Ollama连接成功")
    
    print(f"检查模型: {model_name}")
    if not helper.check_model(model_name):
        print(f"✗ 模型 {model_name} 不存在")
        print(f"正在拉取模型 {model_name}...")
        if helper.pull_model(model_name):
            print(f"✓ 模型 {model_name} 拉取成功")
        else:
            print(f"✗ 模型 {model_name} 拉取失败")
            print(f"请手动拉取: ollama pull {model_name}")
            return False
    else:
        print(f"✓ 模型 {model_name} 已存在")
    
    # 获取模型信息
    info = helper.get_model_info(model_name)
    if info:
        print(f"模型信息: {info.get('modelfile', 'N/A')[:100]}...")
    
    return True

if __name__ == "__main__":
    # 测试脚本
    setup_ollama_model("qwen2.5:32b")

