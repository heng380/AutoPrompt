"""
改写 Agent：根据分析结果改写 prompt
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class RewriteAgent:
    """Agent 3: 根据分析结果改写 prompt"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
    
    def rewrite(self, original_prompt: str, analysis: Dict[str, Any], iteration: int, memory_experiences: str = "") -> str:
        """
        根据分析结果改写 prompt
        
        Args:
            original_prompt: 原始 prompt
            analysis: 分析结果
            iteration: 当前迭代轮次
            memory_experiences: 累积的经验文本
            
        Returns:
            改写后的 prompt
        """
        system_prompt = """你是一个 prompt 优化专家。你的任务是根据分析建议和改进经验来优化 prompt。
请确保：
1. 保留原 prompt 中有效的部分
2. 根据建议进行有针对性的改进
3. 参考历史优化经验，应用通用的优化原则
4. 保持 prompt 的清晰性和可执行性
5. 改进后的 prompt 应该能够解决分析中提到的问题"""
        
        # 构建经验部分的提示
        experience_section = ""
        if memory_experiences and memory_experiences.strip():
            experience_section = f"\n历史优化经验:\n{memory_experiences}\n\n请参考这些经验来改进 prompt。"
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "原始 Prompt:\n{original_prompt}\n\n"
                "分析结果:\n错误数: {error_count}/{total_count}\n\n"
                "改进建议:\n{suggestions}\n\n"
                "{experience_section}"
                "当前迭代轮次: {iteration}\n\n"
                "请根据以上分析建议和历史经验，改写并优化 prompt："
            )
        ]
        
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = chat_prompt.format_messages(
            original_prompt=original_prompt,
            error_count=analysis.get('error_count', 0),
            total_count=analysis.get('total_count', 0),
            suggestions=analysis.get('suggestions', ''),
            experience_section=experience_section,
            iteration=iteration
        )
        
        response = self.llm.invoke(formatted_prompt)
        new_prompt = response.content.strip()
        
        return new_prompt

