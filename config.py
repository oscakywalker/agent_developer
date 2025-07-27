from dataclasses import dataclass

@dataclass
class LLMConfig:
    """LLM配置类"""
    # 选项1: DeepSeek配置
    deepseek_api_key: str = " " #替换实际的API Key
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    
    # 选项2: 千问部署配置
    qwen_api_key: str = " " #替换实际的API Key
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    qwen_model: str = "qwen-plus"
    
    # 连接配置
    timeout: int = 30
    max_retries: int = 2

def load_config() -> LLMConfig:
    """加载配置"""
    return LLMConfig()