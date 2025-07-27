import json
import requests
import logging
import time
import sys
from typing import Dict, Any, List, Optional
from openai import OpenAI

from config import LLMConfig, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模拟天气数据
def get_weather(city: str) -> str:
    weather_data = {
        "beijing": {
            "location": "Beijing",
            "temperature": {"current": 32, "low": 26, "high": 35},
            "rain_probability": 10,
            "humidity": 40
        },
        "shenzhen": {
            "location": "Shenzhen",
            "temperature": {"current": 28, "low": 24, "high": 31},
            "rain_probability": 90,
            "humidity": 85
        }
    }
    city_key = city.lower()
    if city_key in weather_data:
        return json.dumps(weather_data[city_key], ensure_ascii=False)
    return json.dumps({"error": "Weather Unavailable"}, ensure_ascii=False)

AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
}

FUNCTION_DESCRIPTIONS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息，包括温度、降雨概率和湿度",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如beijing或shenzhen"}
            },
            "required": ["city"]
        }
    }
]

#LLM适配器，支持DeepSeek和千问
class LLMAdapter:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.current_api = "auto"  # 'deepseek', 'qwen', or 'auto'
        self.deepseek_client = None
        self.deepseek_available = False
        self.qwen_available = False

        # 尝试初始化DeepSeek客户端
        if config.deepseek_api_key:
            try:
                self.deepseek_client = OpenAI(
                    api_key=config.deepseek_api_key,
                    base_url=config.deepseek_base_url,
                    timeout=config.timeout
                )
                self.deepseek_available = True
                logger.info("DeepSeek客户端初始化成功")
            except Exception as e:
                logger.warning(f"DeepSeek客户端初始化失败: {e}")
                self.deepseek_available = False
        
        # 检查千问API是否配置
        if config.qwen_api_key:
            self.qwen_available = True
            logger.info("千问API配置检查成功")
        else:
            logger.warning("千问API密钥未配置")
            self.qwen_available = False

        # 设置默认API
        if self.current_api == "auto":
            if self.deepseek_available:
                self.current_api = "deepseek"
            elif self.qwen_available:
                self.current_api = "qwen"
            else:
                raise Exception("没有可用的LLM API")

    def set_api(self, api_name: str) -> bool:
        """设置要使用的API，返回是否设置成功"""
        if api_name == "deepseek" and self.deepseek_available:
            self.current_api = "deepseek"
            return True
        elif api_name == "qwen" and self.qwen_available:
            self.current_api = "qwen"
            return True
        elif api_name == "auto":
            # 自动模式，优先使用DeepSeek
            if self.deepseek_available:
                self.current_api = "deepseek"
            elif self.qwen_available:
                self.current_api = "qwen"
            else:
                return False
            return True
        return False

    def get_current_api_info(self) -> Dict[str, Any]:
        """获取当前API的信息"""
        if self.current_api == "deepseek":
            return {
                "name": "DeepSeek",
                "model": self.config.deepseek_model,
                "available": self.deepseek_available
            }
        elif self.current_api == "qwen":
            return {
                "name": "千问",
                "model": self.config.qwen_model,
                "available": self.qwen_available
            }
        else:
            return {"name": "未知", "model": "未知", "available": False}

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用当前选择的LLM API"""
        if self.current_api == "deepseek" and self.deepseek_available:
            try:
                return self._call_deepseek_api(messages)
            except Exception as e:
                logger.warning(f"DeepSeek调用失败: {e}")
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                    self.deepseek_available = False
                    logger.warning("DeepSeek额度已用完，自动切换到千问")
                    if self.qwen_available:
                        self.current_api = "qwen"
                        return self._call_qwen_api(messages)
                raise e

        elif self.current_api == "qwen" and self.qwen_available:
            try:
                return self._call_qwen_api(messages)
            except Exception as e:
                logger.error(f"千问API调用失败: {e}")
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg or "exceeded" in error_msg or "429" in error_msg:
                    self.qwen_available = False
                    logger.warning("千问额度已用完，自动切换到DeepSeek")
                    if self.deepseek_available:
                        self.current_api = "deepseek"
                        return self._call_deepseek_api(messages)
                raise e

        raise Exception("当前没有可用的LLM API")

    def _call_deepseek_api(self, messages: List[Dict[str, str]]) -> str:
        for attempt in range(self.config.max_retries):
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=self.config.deepseek_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"DeepSeek API调用异常 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避策略
                    continue
                else:
                    raise e
        raise Exception("DeepSeek重试次数用尽")

    def _call_qwen_api(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.qwen_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.config.qwen_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        for attempt in range(self.config.max_retries):
            try:
                resp = requests.post(
                    self.config.qwen_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"千问API调用异常 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避策略
                    continue
                else:
                    raise e
        raise Exception("千问API重试次数用尽")

    def _build_function_calling_prompt(self, user_query: str, functions: List[Dict]) -> str:
        function_desc = "可用函数:\n"
        for func in functions:
            function_desc += f"- {func['name']}: {func['description']}\n参数: {json.dumps(func['parameters'], ensure_ascii=False)}\n"
        prompt = f"""{function_desc}

用户请求: {user_query}

请分析用户需求，如果需要调用函数，请按以下格式回复：
FUNCTION_CALL: {{"name": "函数名", "arguments": {{"参数名": "参数值"}}}}

如果不需要调用函数，请直接回答用户问题。
如果已经获得函数结果，请根据结果给出最终建议。

请一步步思考并回答。"""
        return prompt

# 智能体调用核心类
class FunctionCallingAgent:
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter

    def _parse_function_call(self, response: str) -> Optional[Dict]:
        if "FUNCTION_CALL:" in response:
            try:
                func_call_str = response.split("FUNCTION_CALL:")[1].strip()
                func_call_str = func_call_str.split('\n')[0].strip()
                func_call = json.loads(func_call_str)
                return func_call
            except Exception as e:
                logger.warning(f"函数调用解析失败: {e}")
                logger.debug(f"尝试解析的字符串: {response}")
                return None
        return None

    def _execute_function(self, function_call: Dict) -> str:
        func_name = function_call.get("name")
        func_args = function_call.get("arguments", {})
        
        if func_name in AVAILABLE_FUNCTIONS:
            try:
                logger.info(f"执行函数: {func_name}({func_args})")
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)
                logger.info(f"函数结果: {result}")
                return result
            except Exception as e:
                logger.error(f"函数执行失败: {e}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        else:
            logger.warning(f"未知函数: {func_name}")
            return json.dumps({"error": f"未知函数: {func_name}"}, ensure_ascii=False)

    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        logger.info(f"用户查询: {user_query}")
        try:
            initial_prompt = self.llm._build_function_calling_prompt(user_query, FUNCTION_DESCRIPTIONS)
            messages = [{"role": "user", "content": initial_prompt}]
            response = self.llm.call_llm(messages)
            logger.info(f"模型初始响应: {response}")
            function_call = self._parse_function_call(response)

            if function_call:
                function_result = self._execute_function(function_call)
                follow_up_prompt = f"""之前的对话:
用户请求: {user_query}
你决定调用函数: {json.dumps(function_call, ensure_ascii=False)}
函数返回结果: {function_result}

现在请根据函数返回的结果，给出最终的建议回答。请用简洁的语言回答，直接给出有用的信息。"""
            
                messages = [{"role": "user", "content": follow_up_prompt}]
                final_response = self.llm.call_llm(messages)
                logger.info(f"模型最终响应: {final_response}")
                return final_response
            else:
                return response
        except Exception as e:
            logger.error(f"处理查询错误: {e}", exc_info=True)
            return f"抱歉，处理请求时出错: {e}"

#主程序入口
def show_menu(deepseek_available: bool, qwen_available: bool) -> str:
    """显示主菜单"""
    print("\n" + "=" * 50)
    print("函数调用智能体 - API选择")
    print("=" * 50)
    
    options = []
    
    if deepseek_available:
        options.append(("1", "使用 DeepSeek API"))
    
    if qwen_available:
        options.append(("2", "使用 千问 API"))
    
    if deepseek_available and qwen_available:
        options.append(("3", "自动模式 (优先DeepSeek，失败时切换)"))
    
    options.append(("exit", "退出程序"))
    
    for option, desc in options:
        print(f"{option}. {desc}")
    
    while True:
        choice = input("\n请选择模式: ").strip().lower()
        valid_choices = [option.lower() for option, _ in options]
        if choice in valid_choices:
            return choice
        print("无效输入，请重新选择")

def show_api_menu(deepseek_available: bool, qwen_available: bool) -> str:
    """显示API切换菜单"""
    print("\n" + "-" * 30)
    print("API切换菜单")
    print("-" * 30)
    
    options = []
    
    if deepseek_available:
        options.append(("1", "切换到 DeepSeek API"))
    
    if qwen_available:
        options.append(("2", "切换到 千问 API"))
    
    if deepseek_available and qwen_available:
        options.append(("3", "自动模式"))

    options.append(("exit", "退出程序"))
    
    for option, desc in options:
        print(f"{option}. {desc}")
    
    while True:
        choice = input("\n请选择: ").strip().lower()
        valid_choices = [option.lower() for option, _ in options]
        if choice in valid_choices:
            return choice
        print("无效输入，请重新选择")

def main():
    print("=" * 60)
    print("本程序支持在DeepSeek和千问API之间灵活切换，构建一个基础的函数调用的智能体")
    print("=" * 60)
    config = load_config()
    
    try:
        llm_adapter = LLMAdapter(config)
        choice = show_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
        if choice == "exit":
            print("再见！")
            sys.exit(0)
        
        # 根据用户选择设置API
        if choice == "1":
            llm_adapter.set_api("deepseek")
        elif choice == "2":
            llm_adapter.set_api("qwen")
        else:  # 自动模式
            llm_adapter.set_api("auto")

        agent = FunctionCallingAgent(llm_adapter)
        
        while True:
            api_info = llm_adapter.get_current_api_info()
            print(f"\n当前使用: {api_info['name']} ({api_info['model']})")

            query = input("\n请输入您的问题（特殊输入为'switch'-切换API，'exit'-退出程序）: ").strip()

            if query.lower() == 'exit':
                print("再见！")
                break
            elif query.lower() == 'switch':
                api_choice = show_api_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
                if api_choice == "exit":
                    print("再见！")
                    sys.exit(0)
                elif api_choice == "1":
                    success = llm_adapter.set_api("deepseek")
                    print("已切换到DeepSeek API" if success else "切换失败，DeepSeek API不可用")
                elif api_choice == "2":
                    success = llm_adapter.set_api("qwen")
                    print("已切换到千问 API" if success else "切换失败，千问 API不可用")
                else:
                    llm_adapter.set_api("auto")
                    api_info = llm_adapter.get_current_api_info()
                    print(f"已切换到自动模式，当前使用: {api_info['name']}")
                continue
            
            if not query:
                print("请输入有效内容")
                continue

            print("处理中，请稍候...")
            try:
                answer = agent.process_query(query)
                print(f"\n回答: {answer}")
                
                # 每次回答后询问后续操作
                print("\n操作选项:")
                print("1. 继续提问")
                print("2. 切换API")
                print("exit: 退出程序")
                
                while True:
                    op_choice = input("请选择操作 [1/2/exit]: ").strip().lower()
                    if op_choice == '1':
                        break
                    elif op_choice == '2':
                        api_choice = show_api_menu(llm_adapter.deepseek_available, llm_adapter.qwen_available)
                        if api_choice == "exit":
                            print("再见！")
                            sys.exit(0)
                        elif api_choice == "1":
                            success = llm_adapter.set_api("deepseek")
                            print("已切换到DeepSeek API" if success else "切换失败，DeepSeek API不可用")
                        elif api_choice == "2":
                            success = llm_adapter.set_api("qwen")
                            print("已切换到千问 API" if success else "切换失败，千问 API不可用")
                        else:  # 自动模式
                            llm_adapter.set_api("auto")
                            api_info = llm_adapter.get_current_api_info()
                            print(f"已切换到自动模式，当前使用: {api_info['name']}")
                        break
                    elif op_choice == 'exit':
                        print("再见！")
                        sys.exit(0)
                    else:
                        print("无效输入，请重新选择")
                
            except Exception as e:
                print(f"处理请求时出错: {e}")
                
    except Exception as e:
        print(f"程序初始化失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
