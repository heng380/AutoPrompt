"""
配置管理：支持 OpenAI 和 Azure OpenAI
"""
import os
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI


def get_llm(model_name: str = "gpt-3.5-turbo", temperature: float = 0.7) -> ChatOpenAI:
    """
    根据环境变量配置返回合适的 LLM 实例
    
    Args:
        model_name: 模型名称（对于 Azure，这是部署名称）
        temperature: 温度参数
        
    Returns:
        ChatOpenAI 或 AzureChatOpenAI 实例
    """
    # 检查是否配置了 Azure OpenAI
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', model_name)
    
    # 如果配置了 Azure OpenAI，使用 Azure
    if azure_api_key and azure_endpoint:
        # 清理 endpoint URL（移除末尾斜杠）
        endpoint = azure_endpoint.rstrip('/')
        return AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=endpoint,
            openai_api_key=azure_api_key,
            openai_api_version=azure_api_version,
            temperature=temperature
        )
    
    # 否则使用标准的 OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError(
            "未配置 API Key。请设置 OPENAI_API_KEY 或 AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT"
        )
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=openai_api_key
    )

