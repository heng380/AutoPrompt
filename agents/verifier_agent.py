"""
验证 Agent：验证改写后的 prompt 是否能解决目标 badcase
"""
from typing import Dict, List, Any
from agents.prediction_agent import PredictionAgent


class VerifierAgent:
    """Agent 5: 验证改写后的 prompt 是否能解决目标 badcase"""
    
    def __init__(self):
        self.prediction_agent = PredictionAgent()
    
    def verify(self, new_prompt: str, badcases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证改写后的 prompt 是否能解决目标 badcase
        
        Args:
            new_prompt: 改写后的新 prompt
            badcases: 上一轮预测中的错误案例列表（要解决的目标）
            
        Returns:
            包含验证结果的字典：
            - accepted: bool, 是否接受此次改写
            - solved_count: int, 解决了多少个 badcase
            - total_count: int, 总 badcase 数
            - solve_rate: float, 解决率（0-1）
            - verification_results: List[Dict], 每个 badcase 的验证结果
            - feedback: str, 验证反馈信息
        """
        if not badcases:
            return {
                'accepted': True,
                'solved_count': 0,
                'total_count': 0,
                'solve_rate': 1.0,
                'verification_results': [],
                'feedback': "没有 badcase 需要验证，自动接受改写。"
            }
        
        # 只对 badcase 进行预测
        verification_results = self.prediction_agent.predict(new_prompt, badcases)
        
        # 统计解决了多少个 badcase（预测正确的数量）
        solved_count = sum(1 for r in verification_results if r.get('is_correct', False))
        total_count = len(badcases)
        solve_rate = solved_count / total_count if total_count > 0 else 0.0
        
        # 判断是否接受：解决率 >= 50%
        accepted = solve_rate >= 0.5
        
        # 生成反馈信息
        feedback_lines = [
            f"验证结果：解决了 {solved_count}/{total_count} 个 badcase（解决率 {solve_rate:.1%}）"
        ]
        
        if accepted:
            feedback_lines.append(f"✓ 接受此次改写（解决率 >= 50%）")
        else:
            feedback_lines.append(f"✗ 拒绝此次改写（解决率 < 50%），需要回滚并重新改写")
            # 列出仍然错误的案例
            failed_cases = [r for r in verification_results if not r.get('is_correct', False)]
            if failed_cases:
                feedback_lines.append(f"\n仍然未解决的 badcase 示例（最多3个）：")
                for i, case in enumerate(failed_cases[:3], 1):
                    feedback_lines.append(
                        f"  {i}. 输入: {case.get('input', '')[:50]}...\n"
                        f"     预测: {case.get('prediction', '')}\n"
                        f"     正确答案: {case.get('ground_truth', '')}"
                    )
        
        feedback = "\n".join(feedback_lines)
        
        return {
            'accepted': accepted,
            'solved_count': solved_count,
            'total_count': total_count,
            'solve_rate': solve_rate,
            'verification_results': verification_results,
            'feedback': feedback
        }

