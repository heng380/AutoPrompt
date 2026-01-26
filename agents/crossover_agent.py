"""
杂交 Agent：融合两个 prompt 的优势
"""
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class CrossoverAgent:
    """杂交 Agent：融合两个最强 prompt 的优势"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
    
    def crossover(self, prompt1: str, prompt2: str, verification_result1: Dict[str, Any], verification_result2: Dict[str, Any], badcases: list) -> str:
        """
        融合两个 prompt 的优势，生成杂交后的 prompt
        
        Args:
            prompt1: 第一个 prompt（解决率更高的）
            prompt2: 第二个 prompt（解决率第二高的）
            verification_result1: 第一个 prompt 的验证结果
            verification_result2: 第二个 prompt 的验证结果
            badcases: badcase 列表（用于分析）
            
        Returns:
            杂交后的 prompt
        """
        solve_rate1 = verification_result1.get('solve_rate', 0.0)
        solve_rate2 = verification_result2.get('solve_rate', 0.0)
        solved_count1 = verification_result1.get('solved_count', 0)
        solved_count2 = verification_result2.get('solved_count', 0)
        total_count = verification_result1.get('total_count', 0)
        
        # 分析两个 prompt 各自解决了哪些 badcase
        verification_results1 = verification_result1.get('verification_results', [])
        verification_results2 = verification_result2.get('verification_results', [])
        
        # 找出 prompt1 解决但 prompt2 未解决的 case
        solved_by_1_not_2 = []
        for r1 in verification_results1:
            if r1.get('is_correct', False):
                # 检查 prompt2 是否也解决了这个 case
                input_text = r1.get('input', '')
                solved_by_2 = any(
                    r2.get('input', '') == input_text and r2.get('is_correct', False)
                    for r2 in verification_results2
                )
                if not solved_by_2:
                    solved_by_1_not_2.append(r1)
        
        # 找出 prompt2 解决但 prompt1 未解决的 case
        solved_by_2_not_1 = []
        for r2 in verification_results2:
            if r2.get('is_correct', False):
                # 检查 prompt1 是否也解决了这个 case
                input_text = r2.get('input', '')
                solved_by_1 = any(
                    r1.get('input', '') == input_text and r1.get('is_correct', False)
                    for r1 in verification_results1
                )
                if not solved_by_1:
                    solved_by_2_not_1.append(r2)
        
        # 构建分析信息
        analysis_section = f"""
【Prompt 1 的表现】
- 解决率: {solve_rate1:.1%} ({solved_count1}/{total_count})
- 解决的 badcase 数量: {solved_count1}

【Prompt 2 的表现】
- 解决率: {solve_rate2:.1%} ({solved_count2}/{total_count})
- 解决的 badcase 数量: {solved_count2}
"""
        
        if solved_by_1_not_2:
            examples_1 = "\n".join([
                f"  - 输入: {case.get('input', '')[:80]}... (预测: {case.get('prediction', '')} ✓)"
                for case in solved_by_1_not_2[:3]
            ])
            if len(solved_by_1_not_2) > 3:
                examples_1 += f"\n  ... 还有 {len(solved_by_1_not_2) - 3} 个案例"
            analysis_section += f"""
【Prompt 1 独有优势】（共 {len(solved_by_1_not_2)} 个）
Prompt 1 解决了但 Prompt 2 未解决的 badcase 示例：
{examples_1}
"""
        
        if solved_by_2_not_1:
            examples_2 = "\n".join([
                f"  - 输入: {case.get('input', '')[:80]}... (预测: {case.get('prediction', '')} ✓)"
                for case in solved_by_2_not_1[:3]
            ])
            if len(solved_by_2_not_1) > 3:
                examples_2 += f"\n  ... 还有 {len(solved_by_2_not_1) - 3} 个案例"
            analysis_section += f"""
【Prompt 2 独有优势】（共 {len(solved_by_2_not_1)} 个）
Prompt 2 解决了但 Prompt 1 未解决的 badcase 示例：
{examples_2}
"""
        
        system_prompt = """你是一个 prompt 优化专家。你的任务是通过杂交（crossover）融合两个 prompt 的优势，生成一个更好的 prompt。

**重要要求**：
1. 只输出优化后的 prompt 内容，不要包含任何分析、建议、预期改进等元信息
2. 不要输出"预期改进"、"通过以上改进"等描述性文字
3. 不要输出"改进后的 prompt 如下"、"优化后的 prompt 是"等提示性文字
4. 直接输出完整的、可直接使用的 prompt，就像用户会直接使用它一样

**杂交原则**：
1. 仔细分析两个 prompt 的各自优势
2. 融合两个 prompt 中有效的部分
3. 结合两个 prompt 解决不同 badcase 的策略
4. 确保杂交后的 prompt 能够同时解决两个 prompt 各自能解决的 badcase
5. 保持 prompt 的清晰性和可执行性
6. 不要简单地拼接两个 prompt，而是真正融合它们的优势"""
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "【Prompt 1】（解决率更高）:\n{prompt1}\n\n"
                "【Prompt 2】（解决率第二高）:\n{prompt2}\n\n"
                "{analysis_section}\n\n"
                "请分析两个 prompt 的各自优势，融合它们的特点，生成一个能够同时解决两个 prompt 各自能解决的 badcase 的新 prompt。\n\n"
                "**重要**：只输出优化后的 prompt 内容，不要包含任何分析性文字、预期改进说明或元信息。直接输出完整的 prompt 即可。"
            )
        ]
        
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = chat_prompt.format_messages(
            prompt1=prompt1,
            prompt2=prompt2,
            analysis_section=analysis_section
        )
        
        response = self.llm.invoke(formatted_prompt)
        crossover_prompt = response.content.strip()
        
        # 清理可能的分析性内容
        analysis_keywords = ['预期改进', '通过以上改进', '改进后的效果', '预期效果', '改进目标', '通过以上']
        
        for keyword in analysis_keywords:
            if keyword in crossover_prompt:
                idx = crossover_prompt.find(keyword)
                if idx != -1:
                    before = crossover_prompt[:idx].rstrip()
                    
                    separator_patterns = [
                        '\n---\n',
                        '\n\n---\n',
                        '\n---',
                        '\n\n---'
                    ]
                    
                    last_sep_pos = -1
                    for pattern in separator_patterns:
                        pos = before.rfind(pattern)
                        if pos > last_sep_pos:
                            last_sep_pos = pos
                    
                    if last_sep_pos > 0:
                        after_sep = before[last_sep_pos:]
                        newline_after_sep = after_sep.find('\n', len('---'))
                        if newline_after_sep > 0:
                            crossover_prompt = before[:last_sep_pos + len('---') + newline_after_sep].rstrip()
                        else:
                            crossover_prompt = before[:last_sep_pos + len('---')].rstrip()
                    else:
                        crossover_prompt = before
                    break
        
        # 清理末尾的空行
        crossover_prompt = crossover_prompt.rstrip()
        
        return crossover_prompt
