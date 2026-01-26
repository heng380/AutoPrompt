"""
改写 Agent：根据分析结果改写 prompt
"""
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class RewriteAgent:
    """Agent 3: 根据分析结果改写 prompt"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
        self.model_name = model_name
    
    def rewrite(self, original_prompt: str, analysis: Dict[str, Any], iteration: int, memory_experiences: str = "") -> str:
        """
        根据分析结果改写 prompt
        
        Args:
            original_prompt: 要改写的 prompt（首次迭代时是原始 prompt，重新改写时是回滚后的 prompt）
            analysis: 分析结果
            iteration: 当前迭代轮次
            memory_experiences: 累积的经验文本
            
        Returns:
            改写后的 prompt
        """
        system_prompt = """你是一个 prompt 优化专家。你的任务是根据分析建议和改进经验来优化 prompt。

**重要要求**：
1. 只输出优化后的 prompt 内容，不要包含任何分析、建议、预期改进等元信息
2. 不要输出"预期改进"、"通过以上改进"等描述性文字
3. 不要输出"改进后的 prompt 如下"、"优化后的 prompt 是"等提示性文字
4. 直接输出完整的、可直接使用的 prompt，就像用户会直接使用它一样

**优化原则**：
1. 保留原 prompt 中有效的部分
2. 根据建议进行有针对性的改进
3. 参考历史优化经验，应用通用的优化原则
4. 保持 prompt 的清晰性和可执行性
5. 改进后的 prompt 应该能够解决分析中提到的问题
6. 如果有验证反馈，说明上一次改写未能通过验证（解决率低于50%），需要重点改进"""
        
        # 构建经验部分的提示
        experience_section = ""
        if memory_experiences and memory_experiences.strip():
            experience_section = f"\n历史优化经验:\n{memory_experiences}\n\n请参考这些经验来改进 prompt。"
        
        # 构建验证反馈部分
        verification_section = ""
        if analysis.get('verification_feedback'):
            verification_feedback = analysis.get('verification_feedback', '')
            verification_result = analysis.get('verification_result', {})
            solve_rate = verification_result.get('solve_rate', 0.0)
            verification_threshold = analysis.get('verification_threshold', 0.5)  # 默认 50%
            threshold_percent = verification_threshold * 100
            verification_section = (
                f"\n【重要】验证反馈（上一次改写未通过验证）:\n{verification_feedback}\n\n"
                f"上一次改写只解决了 {solve_rate:.1%} 的 badcase，低于 {threshold_percent:.0f}% 的要求。"
                f"请仔细分析验证反馈中仍然未解决的 badcase，重点改进这些方面，"
                f"确保新改写的 prompt 能够解决至少 {threshold_percent:.0f}% 的 badcase。\n\n"
            )
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "原始 Prompt:\n{original_prompt}\n\n"
                "分析结果:\n错误数: {error_count}/{total_count}\n\n"
                "改进建议:\n{suggestions}\n\n"
                "{verification_section}"
                "{experience_section}"
                "当前迭代轮次: {iteration}\n\n"
                "请根据以上分析建议、验证反馈（如有）和历史经验，改写并优化 prompt。\n\n"
                "**重要**：只输出优化后的 prompt 内容，不要包含任何分析性文字、预期改进说明或元信息。直接输出完整的 prompt 即可。"
            )
        ]
        
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = chat_prompt.format_messages(
            original_prompt=original_prompt,
            error_count=analysis.get('error_count', 0),
            total_count=analysis.get('total_count', 0),
            suggestions=analysis.get('suggestions', ''),
            verification_section=verification_section,
            experience_section=experience_section,
            iteration=iteration
        )
        
        response = self.llm.invoke(formatted_prompt)
        new_prompt = response.content.strip()
        
        # 清理可能的分析性内容：移除"预期改进"、"通过以上改进"等段落
        # 这些内容不应该出现在最终的 prompt 中
        analysis_keywords = ['预期改进', '通过以上改进', '改进后的效果', '预期效果', '改进目标', '通过以上']
        
        # 如果 prompt 中包含分析性关键词，移除该关键词及之后的所有内容
        for keyword in analysis_keywords:
            if keyword in new_prompt:
                # 找到关键词的位置
                idx = new_prompt.find(keyword)
                if idx != -1:
                    # 找到该关键词之前的内容
                    before = new_prompt[:idx].rstrip()
                    
                    # 向上查找最后一个分隔线（---），如果存在则保留到分隔线处
                    # 查找各种可能的分隔线格式
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
                        # 找到分隔线，保留到分隔线结束
                        # 查找分隔线后的换行
                        after_sep = before[last_sep_pos:]
                        newline_after_sep = after_sep.find('\n', len('---'))
                        if newline_after_sep > 0:
                            # 保留到分隔线后的第一个换行
                            new_prompt = before[:last_sep_pos + len('---') + newline_after_sep].rstrip()
                        else:
                            # 如果分隔线后没有换行，保留到分隔线位置
                            new_prompt = before[:last_sep_pos + len('---')].rstrip()
                    else:
                        # 如果没有找到分隔线，直接截断到关键词前
                        new_prompt = before
                    
                    # 只处理第一个找到的关键词
                    break
        
        # 清理末尾的空行
        new_prompt = new_prompt.rstrip()
        
        return new_prompt
    
    def rewrite_multiple(self, original_prompt: str, analysis: Dict[str, Any], iteration: int, memory_experiences: str = "", n: int = 1) -> List[str]:
        """
        生成 n 个版本的 prompt（遗传算法候选）
        
        Args:
            original_prompt: 要改写的 prompt（首次迭代时是原始 prompt，重新改写时是回滚后的 prompt）
            analysis: 分析结果
            iteration: 当前迭代轮次
            memory_experiences: 累积的经验文本
            n: 要生成的 prompt 版本数量
            
        Returns:
            n 个改写后的 prompt 列表
        """
        if n <= 1:
            # 如果只需要一个版本，直接调用 rewrite
            return [self.rewrite(original_prompt, analysis, iteration, memory_experiences)]
        
        prompts = []
        # 使用不同的 temperature 来增加多样性
        # temperature 范围：0.5 到 1.2，均匀分布
        base_temperature = 0.5
        max_temperature = 1.2
        
        for i in range(n):
            # 计算当前版本的 temperature
            if n == 1:
                temperature = 0.7
            else:
                temperature = base_temperature + (max_temperature - base_temperature) * (i / (n - 1))
            
            # 为每个版本创建独立的 LLM 实例
            llm = get_llm(model_name=self.model_name, temperature=temperature)
            
            # 使用相同的 prompt 模板，但通过不同的 temperature 产生多样性
            system_prompt = """你是一个 prompt 优化专家。你的任务是根据分析建议和改进经验来优化 prompt。

**重要要求**：
1. 只输出优化后的 prompt 内容，不要包含任何分析、建议、预期改进等元信息
2. 不要输出"预期改进"、"通过以上改进"等描述性文字
3. 不要输出"改进后的 prompt 如下"、"优化后的 prompt 是"等提示性文字
4. 直接输出完整的、可直接使用的 prompt，就像用户会直接使用它一样

**优化原则**：
1. 保留原 prompt 中有效的部分
2. 根据建议进行有针对性的改进
3. 参考历史优化经验，应用通用的优化原则
4. 保持 prompt 的清晰性和可执行性
5. 改进后的 prompt 应该能够解决分析中提到的问题
6. 如果有验证反馈，说明上一次改写未能通过验证（解决率低于50%），需要重点改进
7. 尝试不同的优化策略和表达方式，创造多样化的改进方案"""
            
            # 构建经验部分的提示
            experience_section = ""
            if memory_experiences and memory_experiences.strip():
                experience_section = f"\n历史优化经验:\n{memory_experiences}\n\n请参考这些经验来改进 prompt。"
            
            # 构建验证反馈部分
            verification_section = ""
            if analysis.get('verification_feedback'):
                verification_feedback = analysis.get('verification_feedback', '')
                verification_result = analysis.get('verification_result', {})
                solve_rate = verification_result.get('solve_rate', 0.0)
                verification_threshold = analysis.get('verification_threshold', 0.5)  # 默认 50%
                threshold_percent = verification_threshold * 100
                verification_section = (
                    f"\n【重要】验证反馈（上一次改写未通过验证）:\n{verification_feedback}\n\n"
                    f"上一次改写只解决了 {solve_rate:.1%} 的 badcase，低于 {threshold_percent:.0f}% 的要求。"
                    f"请仔细分析验证反馈中仍然未解决的 badcase，重点改进这些方面，"
                    f"确保新改写的 prompt 能够解决至少 {threshold_percent:.0f}% 的 badcase。\n\n"
                )
            
            messages = [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "原始 Prompt:\n{original_prompt}\n\n"
                    "分析结果:\n错误数: {error_count}/{total_count}\n\n"
                    "改进建议:\n{suggestions}\n\n"
                    "{verification_section}"
                    "{experience_section}"
                    "当前迭代轮次: {iteration}\n\n"
                    "请根据以上分析建议、验证反馈（如有）和历史经验，改写并优化 prompt。\n\n"
                    "**重要**：只输出优化后的 prompt 内容，不要包含任何分析性文字、预期改进说明或元信息。直接输出完整的 prompt 即可。"
                )
            ]
            
            chat_prompt = ChatPromptTemplate.from_messages(messages)
            formatted_prompt = chat_prompt.format_messages(
                original_prompt=original_prompt,
                error_count=analysis.get('error_count', 0),
                total_count=analysis.get('total_count', 0),
                suggestions=analysis.get('suggestions', ''),
                verification_section=verification_section,
                experience_section=experience_section,
                iteration=iteration
            )
            
            response = llm.invoke(formatted_prompt)
            new_prompt = response.content.strip()
            
            # 清理可能的分析性内容：移除"预期改进"、"通过以上改进"等段落
            analysis_keywords = ['预期改进', '通过以上改进', '改进后的效果', '预期效果', '改进目标', '通过以上']
            
            for keyword in analysis_keywords:
                if keyword in new_prompt:
                    idx = new_prompt.find(keyword)
                    if idx != -1:
                        before = new_prompt[:idx].rstrip()
                        
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
                                new_prompt = before[:last_sep_pos + len('---') + newline_after_sep].rstrip()
                            else:
                                new_prompt = before[:last_sep_pos + len('---')].rstrip()
                        else:
                            new_prompt = before
                        break
            
            # 清理末尾的空行
            new_prompt = new_prompt.rstrip()
            prompts.append(new_prompt)
        
        return prompts

