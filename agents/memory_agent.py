"""
记忆 Agent：从 prompt 更改和准确率变化中总结经验
"""
import os
import difflib
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config import get_llm


class MemoryAgent:
    """Agent 4: 从 prompt 更改和准确率变化中总结经验"""
    
    MEMORY_FILE = "memory_experiences.txt"
    EXPERIMENTS_DIR = "experiments"
    
    def __init__(self, experiment_id: str = None, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = get_llm(model_name=model_name, temperature=temperature)
        # 获取项目根目录（agents的父目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # 如果提供了 experiment_id，创建独立文件夹
        if experiment_id:
            self.experiment_id = experiment_id
            self.experiment_dir = os.path.join(project_root, self.EXPERIMENTS_DIR, experiment_id)
            os.makedirs(self.experiment_dir, exist_ok=True)
            self.memory_file_path = os.path.join(self.experiment_dir, self.MEMORY_FILE)
        else:
            # 向后兼容：如果没有提供 experiment_id，使用原来的方式
            self.experiment_id = None
            self.experiment_dir = None
            self.memory_file_path = os.path.join(project_root, self.MEMORY_FILE)
    
    def compute_diff(self, old_prompt: str, new_prompt: str) -> str:
        """计算两个 prompt 之间的差异"""
        if old_prompt == new_prompt:
            return "无变化"
        
        old_lines = old_prompt.splitlines(keepends=True)
        new_lines = new_prompt.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines, 
            new_lines, 
            fromfile='旧prompt', 
            tofile='新prompt',
            lineterm='',
            n=3
        )
        
        return ''.join(diff)
    
    def analyze_case_changes(self, old_results: List[Dict[str, Any]], new_results: List[Dict[str, Any]]) -> str:
        """
        分析预测结果的变化
        
        Args:
            old_results: 上一轮的预测结果
            new_results: 当前轮的预测结果
            
        Returns:
            格式化的case变化描述
        """
        # 找出改进的case（从错误变正确）
        improved_cases = []
        # 找出退化的case（从正确变错误）
        degraded_cases = []
        # 找出预测内容改变但正确性不变的case
        changed_predictions = []
        
        for i, (old_result, new_result) in enumerate(zip(old_results, new_results)):
            old_correct = old_result.get('is_correct', False)
            new_correct = new_result.get('is_correct', False)
            old_pred = str(old_result.get('prediction', ''))
            new_pred = str(new_result.get('prediction', ''))
            input_text = str(old_result.get('input', ''))
            ground_truth = str(old_result.get('ground_truth', ''))
            
            if not old_correct and new_correct:
                # 从错误变正确
                improved_cases.append({
                    'index': i + 1,
                    'input': input_text,
                    'old_prediction': old_pred,
                    'new_prediction': new_pred,
                    'ground_truth': ground_truth
                })
            elif old_correct and not new_correct:
                # 从正确变错误
                degraded_cases.append({
                    'index': i + 1,
                    'input': input_text,
                    'old_prediction': old_pred,
                    'new_prediction': new_pred,
                    'ground_truth': ground_truth
                })
            elif old_pred != new_pred:
                # 预测内容改变但正确性不变
                changed_predictions.append({
                    'index': i + 1,
                    'input': input_text,
                    'old_prediction': old_pred,
                    'new_prediction': new_pred,
                    'ground_truth': ground_truth,
                    'is_correct': new_correct
                })
        
        # 格式化输出
        case_changes_text = []
        
        if improved_cases:
            case_changes_text.append(f"改进的案例（从错误变为正确，共{len(improved_cases)}个）：")
            for case in improved_cases[:5]:  # 最多显示5个
                input_str = case['input'][:100] + "..." if len(case['input']) > 100 else case['input']
                case_changes_text.append(
                    f"  案例{case['index']}:\n"
                    f"    输入: {input_str}\n"
                    f"    修改前预测: {case['old_prediction']}\n"
                    f"    修改后预测: {case['new_prediction']}\n"
                    f"    正确答案: {case['ground_truth']}"
                )
            if len(improved_cases) > 5:
                case_changes_text.append(f"  ... 还有{len(improved_cases) - 5}个改进的案例")
            case_changes_text.append("")
        
        if degraded_cases:
            case_changes_text.append(f"退化的案例（从正确变为错误，共{len(degraded_cases)}个）：")
            for case in degraded_cases[:5]:  # 最多显示5个
                input_str = case['input'][:100] + "..." if len(case['input']) > 100 else case['input']
                case_changes_text.append(
                    f"  案例{case['index']}:\n"
                    f"    输入: {input_str}\n"
                    f"    修改前预测: {case['old_prediction']}\n"
                    f"    修改后预测: {case['new_prediction']}\n"
                    f"    正确答案: {case['ground_truth']}"
                )
            if len(degraded_cases) > 5:
                case_changes_text.append(f"  ... 还有{len(degraded_cases) - 5}个退化的案例")
            case_changes_text.append("")
        
        if changed_predictions and len(changed_predictions) <= 10:
            case_changes_text.append(f"预测内容改变但正确性不变的案例（共{len(changed_predictions)}个）：")
            for case in changed_predictions[:3]:  # 最多显示3个
                input_str = case['input'][:100] + "..." if len(case['input']) > 100 else case['input']
                case_changes_text.append(
                    f"  案例{case['index']}:\n"
                    f"    输入: {input_str}\n"
                    f"    修改前预测: {case['old_prediction']}\n"
                    f"    修改后预测: {case['new_prediction']}\n"
                    f"    正确答案: {case['ground_truth']} ({'正确' if case['is_correct'] else '错误'})"
                )
            if len(changed_predictions) > 3:
                case_changes_text.append(f"  ... 还有{len(changed_predictions) - 3}个预测内容改变的案例")
            case_changes_text.append("")
        
        return "\n".join(case_changes_text) if case_changes_text else "所有案例的预测结果和正确性均未发生变化。"
    
    def summarize_experience(self, prompt_diff: str, acc_change: Dict[str, Any], iteration: int, 
                            case_changes: str = "") -> str:
        """
        总结经验
        
        Args:
            prompt_diff: prompt 更改的 diff
            acc_change: 准确率变化信息，包含 old_acc, new_acc, improvement
            iteration: 当前迭代轮次
            case_changes: case变化的描述文本
            
        Returns:
            总结的经验文本
        """
        old_acc = acc_change.get('old_acc', 0)
        new_acc = acc_change.get('new_acc', 0)
        improvement = acc_change.get('improvement', 0)
        
        # 判断准确率是上升还是下降
        if improvement > 0:
            acc_trend = f"上升了 {improvement:.2%}"
        elif improvement < 0:
            acc_trend = f"下降了 {abs(improvement):.2%}"
        else:
            acc_trend = "保持不变"
        
        system_prompt = """你是一个 prompt 优化经验总结专家。你的任务是从 prompt 的更改、准确率的变化和具体案例的变化中，
提取出通用的、可复用的优化经验和原则。

请按照以下结构总结经验：

1. **迭代轮次**：明确说明是第几轮迭代
2. **更改内容**：详细说明这次 prompt 更改了什么（新增、删除、修改的部分）
3. **准确率变化**：说明准确率是上升了还是下降了，变化幅度是多少
4. **案例变化分析**：分析具体案例的预测变化，说明哪些类型的案例得到了改进，哪些类型的案例出现了退化
5. **后续优化建议**：基于本次更改的效果，给出对后续优化方向的建议

请用清晰、结构化的方式组织内容，重点突出可复用的优化原则。"""
        
        case_section = f"\n案例变化分析:\n{case_changes}\n" if case_changes else ""
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(
                "## 迭代轮次: 第 {iteration} 轮\n\n"
                "## Prompt 更改内容:\n{prompt_diff}\n\n"
                "## 准确率变化:\n"
                "  修改前准确率: {old_acc:.2%}\n"
                "  修改后准确率: {new_acc:.2%}\n"
                "  变化趋势: {acc_trend}\n\n"
                "{case_section}"
                "请根据以上信息，按照要求的结构总结这次优化的经验："
            )
        ]
        
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = chat_prompt.format_messages(
            iteration=iteration,
            prompt_diff=prompt_diff,
            old_acc=old_acc,
            new_acc=new_acc,
            acc_trend=acc_trend,
            case_section=case_section
        )
        
        response = self.llm.invoke(formatted_prompt)
        experience = response.content.strip()
        
        return experience
    
    def save_experience(self, experience: str):
        """将经验保存到文件"""
        os.makedirs(os.path.dirname(self.memory_file_path) or '.', exist_ok=True)
        
        with open(self.memory_file_path, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(experience + "\n")
            f.write("=" * 80 + "\n\n")
    
    def load_experiences(self) -> str:
        """从文件中加载所有经验"""
        if not os.path.exists(self.memory_file_path):
            return ""
        
        try:
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取经验文件出错: {e}")
            return ""
    
    def learn(self, old_prompt: str, new_prompt: str, old_acc: float, new_acc: float, iteration: int,
              old_results: List[Dict[str, Any]] = None, new_results: List[Dict[str, Any]] = None) -> str:
        """
        学习并保存经验
        
        Args:
            old_prompt: 修改前的 prompt
            new_prompt: 修改后的 prompt
            old_acc: 修改前的准确率
            new_acc: 修改后的准确率
            iteration: 当前迭代轮次
            old_results: 上一轮的预测结果
            new_results: 当前轮的预测结果
            
        Returns:
            总结的经验文本
        """
        # 计算 diff
        prompt_diff = self.compute_diff(old_prompt, new_prompt)
        
        # 计算准确率变化
        acc_change = {
            'old_acc': old_acc,
            'new_acc': new_acc,
            'improvement': new_acc - old_acc
        }
        
        # 分析case变化
        case_changes = ""
        if old_results and new_results:
            case_changes = self.analyze_case_changes(old_results, new_results)
        
        # 总结经验
        experience = self.summarize_experience(prompt_diff, acc_change, iteration, case_changes)
        
        # 保存经验
        self.save_experience(experience)
        
        return experience

