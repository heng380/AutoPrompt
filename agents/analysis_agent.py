"""
分析 Agent：分析错误案例并给出修改意见
"""
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class AnalysisAgent:
    """Agent 2: 分析错误案例并给出修改意见"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
    
    def analyze(self, prompt: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析预测结果中的错误案例，给出修改意见
        
        Args:
            prompt: 当前的 prompt
            results: 预测结果列表
            
        Returns:
            包含分析结果的字典
        """
        # 找出错误的案例
        error_cases = [r for r in results if not r.get('is_correct', True)]
        
        if not error_cases:
            return {
                'has_errors': False,
                'error_count': 0,
                'total_count': len(results),
                'suggestions': "所有预测都正确，无需修改。"
            }
        
        # 准备错误案例样本（最多5个）
        error_samples = error_cases[:5]
        error_text = self._format_error_cases(error_samples)
        
        system_prompt = """你是一个 prompt 分析专家。你的任务是分析预测错误的原因，并给出改进 prompt 的具体建议。
请仔细分析错误案例，找出 prompt 中可能存在的问题，如：
1. 指令不够清晰
2. 缺少关键信息
3. 格式要求不明确
4. 示例不足或不够典型
5. 逻辑或推理步骤缺失

请给出具体、可操作的改进建议。"""
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "当前 Prompt:\n{prompt}\n\n错误案例:\n{error_cases}\n\n"
                "总错误数: {error_count}/{total_count}\n\n"
                "请分析这些错误案例，找出 prompt 的问题，并给出具体的改进建议："
            )
        ]
        
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = chat_prompt.format_messages(
            prompt=prompt,
            error_cases=error_text,
            error_count=len(error_cases),
            total_count=len(results)
        )
        
        response = self.llm.invoke(formatted_prompt)
        suggestions = response.content.strip()
        
        return {
            'has_errors': True,
            'error_count': len(error_cases),
            'total_count': len(results),
            'error_cases': error_samples,
            'suggestions': suggestions
        }
    
    def _format_error_cases(self, error_cases: List[Dict[str, Any]]) -> str:
        """格式化错误案例"""
        formatted = []
        for i, case in enumerate(error_cases, 1):
            formatted.append(
                f"案例 {i}:\n"
                f"输入: {case.get('input', '')}\n"
                f"预测: {case.get('prediction', '')}\n"
                f"正确答案: {case.get('ground_truth', '')}\n"
            )
        return "\n".join(formatted)

