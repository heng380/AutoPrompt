"""
测试 OpenAI API 连通性

使用方法：
1. 确保已安装依赖: pip install -r requirements.txt
2. 确保已配置 .env 文件
3. 运行: python test_api.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    from config import get_llm
except ImportError as e:
    print("❌ 缺少依赖包，请先安装：")
    print("   pip install -r requirements.txt")
    print(f"\n错误详情: {e}")
    sys.exit(1)

# 加载环境变量
load_dotenv()

def test_openai_connection():
    """测试 OpenAI API 连接"""
    print("=" * 60)
    print("测试 OpenAI API 连通性")
    print("=" * 60)
    
    # 检查配置
    openai_key = os.getenv('OPENAI_API_KEY')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    print("\n📋 配置检查:")
    if azure_key and azure_endpoint:
        print("✓ 检测到 Azure OpenAI 配置")
        print(f"  - Endpoint: {azure_endpoint}")
        print(f"  - API Key: {'*' * 20}...{azure_key[-4:] if len(azure_key) > 4 else '****'}")
        api_type = "Azure OpenAI"
    elif openai_key:
        print("✓ 检测到 OpenAI 配置")
        print(f"  - API Key: {'*' * 20}...{openai_key[-4:] if len(openai_key) > 4 else '****'}")
        api_type = "OpenAI"
    else:
        print("❌ 未检测到 API Key 配置")
        print("\n请检查 .env 文件，确保配置了以下之一：")
        print("  - OPENAI_API_KEY (标准 OpenAI)")
        print("  - AZURE_OPENAI_API_KEY 和 AZURE_OPENAI_ENDPOINT (Azure OpenAI)")
        return False
    
    # 测试 API 调用
    print(f"\n🔄 测试 {api_type} API 调用...")
    try:
        llm = get_llm(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # 发送一个简单的测试请求
        test_prompt = "请用一句话回答：1+1等于几？只回答数字。"
        print(f"\n📤 发送测试请求...")
        print(f"   提示: {test_prompt}")
        
        response = llm.invoke(test_prompt)
        result = response.content.strip()
        
        print(f"\n✅ API 调用成功！")
        print(f"📥 响应: {result}")
        
        # 验证响应
        if result:
            print(f"\n✓ 连通性测试通过")
            print(f"✓ API 正常工作")
            return True
        else:
            print(f"\n⚠️  收到空响应")
            return False
            
    except Exception as e:
        print(f"\n❌ API 调用失败")
        print(f"错误信息: {str(e)}")
        print(f"\n错误类型: {type(e).__name__}")
        
        # 提供常见错误的解决建议
        error_str = str(e).lower()
        if "api key" in error_str or "authentication" in error_str:
            print("\n💡 建议:")
            print("  - 检查 API Key 是否正确")
            print("  - 确认 API Key 是否有效且未过期")
        elif "endpoint" in error_str or "url" in error_str:
            print("\n💡 建议:")
            print("  - 检查 Azure OpenAI Endpoint 是否正确")
            print("  - 确认 Endpoint URL 格式为: https://your-resource.openai.azure.com/")
        elif "rate limit" in error_str or "quota" in error_str:
            print("\n💡 建议:")
            print("  - 检查 API 配额是否已用完")
            print("  - 等待一段时间后重试")
        elif "model" in error_str or "deployment" in error_str:
            print("\n💡 建议:")
            print("  - 检查模型名称或部署名称是否正确")
            print("  - 确认该模型/部署是否可用")
        
        return False

def test_agents():
    """测试三个 Agent 是否正常工作"""
    print("\n" + "=" * 60)
    print("测试 Agent 初始化")
    print("=" * 60)
    
    try:
        from agents import PredictionAgent, AnalysisAgent, RewriteAgent
        
        print("\n🔄 初始化 PredictionAgent...")
        pred_agent = PredictionAgent()
        print("✓ PredictionAgent 初始化成功")
        
        print("\n🔄 初始化 AnalysisAgent...")
        analysis_agent = AnalysisAgent()
        print("✓ AnalysisAgent 初始化成功")
        
        print("\n🔄 初始化 RewriteAgent...")
        rewrite_agent = RewriteAgent()
        print("✓ RewriteAgent 初始化成功")
        
        print("\n✅ 所有 Agent 初始化成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent 初始化失败")
        print(f"错误信息: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AutoPrompt API 连通性测试")
    print("=" * 60)
    
    # 测试 API 连通性
    api_ok = test_openai_connection()
    
    # 测试 Agent 初始化
    agents_ok = test_agents()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if api_ok and agents_ok:
        print("\n✅ 所有测试通过！系统已准备就绪。")
        print("\n你可以运行以下命令启动应用：")
        print("  python app.py")
    else:
        print("\n❌ 部分测试失败，请检查配置和错误信息。")
        if not api_ok:
            print("  - API 连通性测试失败")
        if not agents_ok:
            print("  - Agent 初始化测试失败")
    
    print("\n" + "=" * 60)

