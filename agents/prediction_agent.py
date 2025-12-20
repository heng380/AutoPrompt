"""
预测 Agent：根据数据集和 prompt 进行预测
"""
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class PredictionAgent:
    """Agent 1: 根据数据集和 prompt 进行预测"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
    
    def predict(self, prompt: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对数据集中的每个样本进行预测
        
        Args:
            prompt: 当前的 prompt
            dataset: 数据集，每个样本包含输入字段
            
        Returns:
            包含预测结果的列表
        """
        results = []
        
        system_prompt = """你是一个预测助手。根据给定的 prompt 和输入数据，生成预测结果。
请严格按照 prompt 的要求进行预测。"""
        
        for item in dataset:
            # 构建输入文本
            input_text = self._format_input(item)
            
            messages = [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Prompt: {prompt}\n\n输入数据: {input_text}\n\n请给出预测结果:"
                )
            ]
            
            chat_prompt = ChatPromptTemplate.from_messages(messages)
            formatted_prompt = chat_prompt.format_messages(
                prompt=prompt,
                input_text=input_text
            )
            
            response = self.llm.invoke(formatted_prompt)
            prediction = response.content.strip()
            
            result = {
                **item,
                'prediction': prediction,
                'ground_truth': item.get('ground_truth', ''),
                'is_correct': self._check_correctness(prediction, item.get('ground_truth', ''))
            }
            results.append(result)
        
        return results
    
    def _format_input(self, item: Dict[str, Any]) -> str:
        """格式化输入数据"""
        # 如果输入是字典，转换为字符串
        if isinstance(item.get('input'), dict):
            return str(item['input'])
        return str(item.get('input', ''))
    
    def _check_correctness(self, prediction: str, ground_truth: str) -> bool:
        """简单的正确性检查（可以根据实际需求改进）"""
        if not ground_truth:
            return True  # 如果没有标准答案，认为正确
        return prediction.strip().lower() == ground_truth.strip().lower()

