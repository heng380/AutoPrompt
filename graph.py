"""
使用 LangGraph 构建的 AutoPrompt 优化工作流
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agents import PredictionAgent, AnalysisAgent, RewriteAgent, MemoryAgent, VerifierAgent, CrossoverAgent
from utils.logger import get_log_collector

# 获取日志收集器
_log_collector = get_log_collector()


class AutoPromptState(TypedDict, total=False):
    """工作流状态定义"""
    original_prompt: str      # 原始 prompt
    current_prompt: str       # 当前 prompt
    dataset: List[Dict[str, Any]]  # 数据集
    results: List[Dict[str, Any]]  # 预测结果
    analysis: Dict[str, Any]  # 分析结果
    iteration: int            # 当前迭代轮次
    max_iterations: int       # 最大迭代轮次
    history: List[Dict[str, Any]]  # 历史记录
    previous_prompt: str      # 上一轮的 prompt（用于计算diff）
    memory_experiences: str   # 累积的经验文本
    verification_result: Dict[str, Any]  # 验证结果
    pending_prompt: str       # 待验证的改写后的 prompt（如果验证失败则回滚）
    verification_threshold: float  # 验证阈值（0-1），解决率 >= threshold 时接受改写
    candidate_prompts: List[str]  # 候选 prompt 列表（遗传算法）
    candidate_count: int     # 遗传算法候选数量
    accepted_candidates: List[Dict[str, Any]]  # 所有通过验证的候选（用于 crossover）


class AutoPromptGraph:
    """AutoPrompt 优化工作流图"""
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id
        self.prediction_agent = PredictionAgent()
        self.analysis_agent = AnalysisAgent()
        self.rewrite_agent = RewriteAgent()
        self.memory_agent = MemoryAgent(experiment_id=experiment_id)
        self.verifier_agent = VerifierAgent()
        self.crossover_agent = CrossoverAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AutoPromptState)
        
        # 添加节点
        workflow.add_node("predict", self._predict_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("verify", self._verify_node)  # 验证节点：验证改写后的 prompt
        workflow.add_node("crossover", self._crossover_node)  # 杂交节点：融合两个最强 prompt
        workflow.add_node("memory", self._memory_node)
        workflow.add_node("final_predict", self._final_predict_node)  # 最后一轮预测节点
        
        # 设置入口点
        workflow.add_edge(START, "predict")
        
        # 添加边
        workflow.add_edge("predict", "memory")  # predict后立即总结经验
        workflow.add_edge("memory", "analyze")  # 然后分析
        
        # 条件边：分析后判断是否继续（如果所有预测都正确，直接结束）
        workflow.add_conditional_edges(
            "analyze",
            self._should_continue_after_analyze,
            {
                "continue": "rewrite",  # 有错误，继续改写
                "end": END  # 所有预测都正确，直接结束
            }
        )
        
        workflow.add_edge("rewrite", "verify")  # 改写后验证
        
        # 添加一个更新迭代次数的节点
        workflow.add_node("increment_iteration", self._increment_iteration_node)
        
        # 添加检查是否继续的辅助节点
        workflow.add_node("check_continue", self.check_continue)
        
        # 条件边：验证节点后，判断是否接受改写
        workflow.add_conditional_edges(
            "verify",
            self._should_accept_rewrite,
            {
                "accept": "crossover",  # 接受改写，进入 crossover 节点判断是否需要杂交
                "reject": "analyze"  # 拒绝改写，回滚后重新分析
            }
        )
        
        # crossover 节点后，直接进入 check_continue
        workflow.add_edge("crossover", "check_continue")
        
        # 条件边：决定是否继续迭代（从 verify 接受后出发）
        workflow.add_conditional_edges(
            "check_continue",
            self._should_continue,
            {
                "continue": "increment_iteration",
                "final_predict": "final_predict",  # 最后一轮，先预测再结束
                "end": END
            }
        )
        
        # 从 increment_iteration 回到 predict
        workflow.add_edge("increment_iteration", "predict")
        
        # 最后一轮预测后结束
        workflow.add_edge("final_predict", END)
        
        return workflow.compile()
    
    def _predict_node(self, state: AutoPromptState) -> AutoPromptState:
        """预测节点：使用当前 prompt 进行预测"""
        # 在新迭代开始时添加分隔线
        separator = "─" * 50 + f" 迭代 {state['iteration']} " + "─" * 50
        _log_collector.log(separator, level="separator")
        print(separator)
        
        _log_collector.log(f"[迭代 {state['iteration']}] [预测] 开始预测...", level="info")
        print(f"[迭代 {state['iteration']}] [预测] 开始预测...")
        
        results = self.prediction_agent.predict(
            state['current_prompt'],
            state['dataset']
        )
        
        # 打印预测结果摘要
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        _log_collector.log(f"[迭代 {state['iteration']}] [预测] 预测完成: {correct_count}/{total_count} 正确", level="success")
        print(f"[迭代 {state['iteration']}] [预测] 预测完成: {correct_count}/{total_count} 正确")
        
        # 打印每个样本的预测结果
        _log_collector.log(f"\n[迭代 {state['iteration']}] [预测] 详细预测结果:", level="info")
        print(f"\n[迭代 {state['iteration']}] [预测] 详细预测结果:")
        for i, result in enumerate(results, 1):
            input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
            prediction = result.get('prediction', '')
            ground_truth = result.get('ground_truth', '')
            is_correct = result.get('is_correct', False)
            status = "✓" if is_correct else "✗"
            level = "success" if is_correct else "error"
            log_msg = f"  {status} [{i}] 输入: {input_text}\n      预测: {prediction} | 正确答案: {ground_truth}"
            _log_collector.log(log_msg, level=level)
            print(f"  {status} [{i}] 输入: {input_text}")
            print(f"      预测: {prediction} | 正确答案: {ground_truth}")
        
        state['results'] = results
        return state
    
    def _analyze_node(self, state: AutoPromptState) -> AutoPromptState:
        """分析节点：分析错误案例"""
        # 检查是否有验证反馈（说明是验证失败后重新分析的）
        has_verification_feedback = state.get('analysis') and state['analysis'].get('verification_feedback')
        
        if has_verification_feedback:
            verification_feedback = state['analysis'].get('verification_feedback')
            verification_result = state['analysis'].get('verification_result')
            _log_collector.log(f"[迭代 {state['iteration']}] [分析] 基于验证反馈重新分析错误案例...", level="warning")
            print(f"[迭代 {state['iteration']}] [分析] 基于验证反馈重新分析错误案例...")
            
            # 先保存验证反馈信息，然后清空 analysis 重新分析
            saved_verification_feedback = verification_feedback
            saved_verification_result = verification_result
        else:
            _log_collector.log(f"[迭代 {state['iteration']}] [分析] 开始分析错误案例...", level="info")
            print(f"[迭代 {state['iteration']}] [分析] 开始分析错误案例...")
            saved_verification_feedback = None
            saved_verification_result = None
        
        # 重新分析（基于当前 prompt 和 results）
        analysis = self.analysis_agent.analyze(
            state['current_prompt'],
            state['results']
        )
        
        # 如果有验证反馈，将其加入到分析结果中（供 rewrite 使用）
        if saved_verification_feedback:
            analysis['verification_feedback'] = saved_verification_feedback
            analysis['verification_result'] = saved_verification_result
            
            # 获取被拒绝的 prompt 和解决了/未解决的 case
            rejected_prompt = saved_verification_result.get('rejected_prompt', '')
            solved_cases = saved_verification_result.get('solved_cases', [])
            failed_cases = saved_verification_result.get('failed_cases', [])
            
            # 构建对比分析信息
            comparison_section = ""
            if rejected_prompt:
                comparison_section += f"\n【被拒绝的改写 prompt】:\n{rejected_prompt}\n\n"
            
            if solved_cases:
                solved_text = "\n".join([
                    f"  - 案例 {i+1}: {case.get('input', '')[:50]}... (预测: {case.get('prediction', '')} ✓)"
                    for i, case in enumerate(solved_cases[:5])  # 最多显示5个
                ])
                if len(solved_cases) > 5:
                    solved_text += f"\n  ... 还有 {len(solved_cases) - 5} 个被解决的案例"
                comparison_section += f"【被解决的 badcase】（共 {len(solved_cases)} 个）:\n{solved_text}\n\n"
            
            if failed_cases:
                failed_text = "\n".join([
                    f"  - 案例 {i+1}: {case.get('input', '')[:50]}... (预测: {case.get('prediction', '')} ✗, 正确答案: {case.get('ground_truth', '')})"
                    for i, case in enumerate(failed_cases[:5])  # 最多显示5个
                ])
                if len(failed_cases) > 5:
                    failed_text += f"\n  ... 还有 {len(failed_cases) - 5} 个未解决的案例"
                comparison_section += f"【未解决的 badcase】（共 {len(failed_cases)} 个）:\n{failed_text}\n\n"
            
            verification_threshold = state.get('verification_threshold', 0.5)
            threshold_percent = verification_threshold * 100
            analysis['suggestions'] = (
                f"【验证反馈】{saved_verification_feedback}\n\n"
                f"{comparison_section}"
                f"【对比分析要求】:\n"
                f"请对比被拒绝的改写 prompt 和当前的 prompt，分析：\n"
                f"1. 为什么这些 badcase 被解决了？被拒绝的 prompt 在哪些方面做得更好？\n"
                f"2. 为什么那些 badcase 仍然未解决？被拒绝的 prompt 在哪些方面还有不足？\n"
                f"3. 如何结合两者的优点，创建一个既解决已解决的 case，又能解决未解决 case 的新 prompt？\n\n"
                f"原改进建议：\n{analysis.get('suggestions', '')}\n\n"
                f"请结合验证反馈和对比分析，重新思考如何改进 prompt，确保能够解决至少 {threshold_percent:.0f}% 的 badcase。"
            )
        
        state['analysis'] = analysis
        
        # 只在首次分析时记录历史（验证失败后重新分析不重复记录）
        if not has_verification_feedback:
            # 记录历史（包含每轮的详细预测结果）
            history_entry = {
                'iteration': state['iteration'],
                'prompt': state['current_prompt'],
                'error_count': analysis.get('error_count', 0),
                'total_count': analysis.get('total_count', 0),
                'suggestions': analysis.get('suggestions', ''),
                'results': state['results']  # 保存每轮的详细预测结果
            }
            state['history'].append(history_entry)
            _log_collector.log(f"[迭代 {state['iteration']}] [系统] 已记录历史，当前 history 长度: {len(state['history'])}", level="info")
            print(f"[迭代 {state['iteration']}] [系统] 已记录历史，当前 history 长度: {len(state['history'])}")
        
        return state
    
    def check_continue(self, state: AutoPromptState) -> AutoPromptState:
        """检查是否继续的辅助节点（接受改写后）"""
        # 这个节点只是传递状态，实际判断在 _should_continue 中
        return state
    
    def _should_continue_after_analyze(self, state: AutoPromptState) -> str:
        """分析后判断是否继续迭代"""
        # 检查是否所有预测都正确
        analysis = state.get('analysis', {})
        if not analysis.get('has_errors', True):
            _log_collector.log(f"[迭代 {state['iteration']}] [系统] 所有预测都正确，结束优化", level="success")
            print(f"[迭代 {state['iteration']}] [系统] 所有预测都正确，结束优化")
            return "end"
        
        # 有错误，继续改写
        return "continue"
    
    def _rewrite_node(self, state: AutoPromptState) -> AutoPromptState:
        """改写节点：根据分析结果改写 prompt（支持生成多个候选版本）"""
        candidate_count = state.get('candidate_count', 1)
        
        if candidate_count > 1:
            _log_collector.log(f"[迭代 {state['iteration']}] [改写] 开始生成 {candidate_count} 个候选 prompt 版本...", level="info")
            print(f"[迭代 {state['iteration']}] [改写] 开始生成 {candidate_count} 个候选 prompt 版本...")
        else:
            _log_collector.log(f"[迭代 {state['iteration']}] [改写] 开始改写 prompt...", level="info")
            print(f"[迭代 {state['iteration']}] [改写] 开始改写 prompt...")
        
        # 保存当前 prompt 作为待验证的 prompt（如果验证失败可以回滚）
        state['pending_prompt'] = state['current_prompt']
        
        # 获取已有的经验
        memory_experiences = state.get('memory_experiences', '')
        if not memory_experiences:
            memory_experiences = self.memory_agent.load_experiences()
            state['memory_experiences'] = memory_experiences
        
        # 将验证阈值信息传递给 rewrite agent（用于生成提示信息）
        verification_threshold = state.get('verification_threshold', 1.0)
        if state.get('analysis'):
            state['analysis']['verification_threshold'] = verification_threshold
        
        # 生成候选 prompt 列表
        # 使用 current_prompt 而不是 original_prompt，这样重新改写时基于回滚后的 prompt
        candidate_prompts = self.rewrite_agent.rewrite_multiple(
            state['current_prompt'],
            state['analysis'],
            state['iteration'],
            memory_experiences,
            n=candidate_count
        )
        
        state['candidate_prompts'] = candidate_prompts
        
        if candidate_count > 1:
            _log_collector.log(f"[迭代 {state['iteration']}] [改写] 已生成 {len(candidate_prompts)} 个候选版本，准备验证...", level="success")
            print(f"[迭代 {state['iteration']}] [改写] 已生成 {len(candidate_prompts)} 个候选版本，准备验证...")
        else:
            # 如果只有一个版本，直接设置为 current_prompt
            state['current_prompt'] = candidate_prompts[0]
        
        return state
    
    def _verify_node(self, state: AutoPromptState) -> AutoPromptState:
        """验证节点：验证改写后的 prompt 是否能解决目标 badcase（支持多个候选版本）"""
        candidate_prompts = state.get('candidate_prompts', [])
        candidate_count = state.get('candidate_count', 1)
        
        # 获取当前轮次的 badcase（错误案例）
        badcases = []
        analysis = state.get('analysis', {})
        if analysis.get('has_errors', False):
            badcases = [r for r in state.get('results', []) if not r.get('is_correct', True)]
        
        if not badcases:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 没有 badcase 需要验证，自动接受改写", level="info")
            print(f"[迭代 {state['iteration']}] [验证] 没有 badcase 需要验证，自动接受改写")
            # 如果有多个候选，选择第一个；否则使用 current_prompt
            if candidate_prompts:
                state['current_prompt'] = candidate_prompts[0]
            state['verification_result'] = {
                'accepted': True,
                'solved_count': 0,
                'total_count': 0,
                'solve_rate': 1.0,
                'verification_results': [],
                'feedback': "没有 badcase 需要验证，自动接受改写。"
            }
            return state
        
        verification_threshold = state.get('verification_threshold', 1.0)
        
        # 如果有多个候选版本，评估所有版本并选择最好的
        if candidate_count > 1 and candidate_prompts:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 开始评估 {len(candidate_prompts)} 个候选版本，需要验证 {len(badcases)} 个 badcase...", level="info")
            print(f"[迭代 {state['iteration']}] [验证] 开始评估 {len(candidate_prompts)} 个候选版本，需要验证 {len(badcases)} 个 badcase...")
            
            # 评估所有候选版本
            candidate_results = []
            for i, candidate_prompt in enumerate(candidate_prompts, 1):
                _log_collector.log(f"[迭代 {state['iteration']}] [验证] 评估候选版本 {i}/{len(candidate_prompts)}...", level="info")
                print(f"[迭代 {state['iteration']}] [验证] 评估候选版本 {i}/{len(candidate_prompts)}...")
                
                verification_result = self.verifier_agent.verify(
                    new_prompt=candidate_prompt,
                    badcases=badcases,
                    threshold=verification_threshold
                )
                
                candidate_results.append({
                    'prompt': candidate_prompt,
                    'verification_result': verification_result,
                    'solve_rate': verification_result.get('solve_rate', 0.0),
                    'solved_count': verification_result.get('solved_count', 0),
                    'total_count': verification_result.get('total_count', 0),
                    'accepted': verification_result.get('accepted', False)
                })
            
            # 选择最好的候选版本（优先选择 accepted 的，如果都 accepted 则选择解决率最高的，否则选择解决率最高的）
            best_candidate = None
            best_index = 0
            
            # 先找 accepted 的版本中解决率最高的
            accepted_candidates_list = [c for c in candidate_results if c['accepted']]
            if accepted_candidates_list:
                best_candidate = max(accepted_candidates_list, key=lambda x: x['solve_rate'])
                best_index = candidate_results.index(best_candidate)
            else:
                # 如果没有 accepted 的，选择解决率最高的
                best_candidate = max(candidate_results, key=lambda x: x['solve_rate'])
                best_index = candidate_results.index(best_candidate)
            
            # 设置选中的 prompt 为 current_prompt
            state['current_prompt'] = best_candidate['prompt']
            state['verification_result'] = best_candidate['verification_result']
            
            # 保存所有 accepted 的候选（用于 crossover）
            # 按解决率排序，取前两个最强的
            if len(accepted_candidates_list) >= 2:
                sorted_accepted = sorted(accepted_candidates_list, key=lambda x: x['solve_rate'], reverse=True)
                state['accepted_candidates'] = sorted_accepted[:2]  # 保存最强的两个
            elif len(accepted_candidates_list) == 1:
                state['accepted_candidates'] = accepted_candidates_list  # 只有一个，也保存（但不会进行 crossover）
            else:
                state['accepted_candidates'] = []  # 没有 accepted 的
            
            # 添加遗传算法选择信息
            state['verification_result']['genetic_algorithm'] = {
                'candidate_count': len(candidate_prompts),
                'selected_index': best_index + 1,
                'accepted_count': len(accepted_candidates_list),
                'all_candidates': [
                    {
                        'index': i + 1,
                        'solve_rate': c['solve_rate'],
                        'solved_count': c['solved_count'],
                        'total_count': c['total_count'],
                        'accepted': c['accepted']
                    }
                    for i, c in enumerate(candidate_results)
                ]
            }
            
            # 打印所有候选版本的结果
            _log_collector.log(f"\n[迭代 {state['iteration']}] [验证] 所有候选版本评估结果:", level="info")
            print(f"\n[迭代 {state['iteration']}] [验证] 所有候选版本评估结果:")
            for i, candidate_result in enumerate(candidate_results, 1):
                solve_rate = candidate_result['solve_rate']
                solved_count = candidate_result['solved_count']
                total_count = candidate_result['total_count']
                accepted = candidate_result['accepted']
                selected_mark = "✓ 选中" if i == best_index + 1 else ""
                status = "✓" if accepted else "✗"
                log_msg = f"  版本 {i}: {status} 解决了 {solved_count}/{total_count} 个 badcase（解决率 {solve_rate:.1%}）{selected_mark}"
                level = "success" if i == best_index + 1 else "info"
                _log_collector.log(log_msg, level=level)
                print(f"  版本 {i}: {status} 解决了 {solved_count}/{total_count} 个 badcase（解决率 {solve_rate:.1%}）{selected_mark}")
            
            # 打印选中版本的详细验证结果
            verification_results = best_candidate['verification_result'].get('verification_results', [])
            if verification_results:
                _log_collector.log(f"\n[迭代 {state['iteration']}] [验证] 选中版本 {best_index + 1} 的详细验证结果:", level="info")
                print(f"\n[迭代 {state['iteration']}] [验证] 选中版本 {best_index + 1} 的详细验证结果:")
                for i, result in enumerate(verification_results, 1):
                    input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
                    prediction = result.get('prediction', '')
                    ground_truth = result.get('ground_truth', '')
                    is_correct = result.get('is_correct', False)
                    status = "✓" if is_correct else "✗"
                    level = "success" if is_correct else "error"
                    log_msg = f"  {status} [{i}] 输入: {input_text}\n      预测: {prediction} | 正确答案: {ground_truth}"
                    _log_collector.log(log_msg, level=level)
                    print(f"  {status} [{i}] 输入: {input_text}")
                    print(f"      预测: {prediction} | 正确答案: {ground_truth}")
            
            # 记录验证反馈到日志
            feedback_level = "success" if best_candidate['accepted'] else "warning"
            _log_collector.log(best_candidate['verification_result'].get('feedback', ''), level=feedback_level)
            
        else:
            # 单个版本的情况（原有逻辑）
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 开始验证改写后的 prompt...", level="info")
            print(f"[迭代 {state['iteration']}] [验证] 开始验证改写后的 prompt...")
            
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 需要验证 {len(badcases)} 个 badcase...", level="info")
            print(f"[迭代 {state['iteration']}] [验证] 需要验证 {len(badcases)} 个 badcase...")
            
            # 使用改写后的 prompt 验证 badcase
            verification_result = self.verifier_agent.verify(
                new_prompt=state['current_prompt'],
                badcases=badcases,
                threshold=verification_threshold
            )
            
            # 打印验证结果摘要
            solve_rate = verification_result.get('solve_rate', 0.0)
            solved_count = verification_result.get('solved_count', 0)
            total_count = verification_result.get('total_count', 0)
            accepted = verification_result.get('accepted', False)
            
            _log_collector.log(f"\n[迭代 {state['iteration']}] [验证] 验证结果: 解决了 {solved_count}/{total_count} 个 badcase（解决率 {solve_rate:.1%}）", level="info")
            print(f"\n[迭代 {state['iteration']}] [验证] 验证结果: 解决了 {solved_count}/{total_count} 个 badcase（解决率 {solve_rate:.1%}）")
            
            # 打印每个 badcase 的验证结果（格式与 predict 一致）
            verification_results = verification_result.get('verification_results', [])
            if verification_results:
                _log_collector.log(f"\n[迭代 {state['iteration']}] [验证] 详细验证结果:", level="info")
                print(f"\n[迭代 {state['iteration']}] [验证] 详细验证结果:")
                for i, result in enumerate(verification_results, 1):
                    input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
                    prediction = result.get('prediction', '')
                    ground_truth = result.get('ground_truth', '')
                    is_correct = result.get('is_correct', False)
                    status = "✓" if is_correct else "✗"
                    level = "success" if is_correct else "error"
                    log_msg = f"  {status} [{i}] 输入: {input_text}\n      预测: {prediction} | 正确答案: {ground_truth}"
                    _log_collector.log(log_msg, level=level)
                    print(f"  {status} [{i}] 输入: {input_text}")
                    print(f"      预测: {prediction} | 正确答案: {ground_truth}")
            
            # 记录验证反馈到日志
            feedback_level = "success" if accepted else "warning"
            _log_collector.log(verification_result.get('feedback', ''), level=feedback_level)
            
            state['verification_result'] = verification_result
            
            # 单个版本的情况，如果有 accepted，也保存到 accepted_candidates
            if accepted:
                state['accepted_candidates'] = [{
                    'prompt': state['current_prompt'],
                    'verification_result': verification_result,
                    'solve_rate': solve_rate,
                    'solved_count': solved_count,
                    'total_count': total_count,
                    'accepted': True
                }]
            else:
                state['accepted_candidates'] = []
        
        return state
    
    def _should_accept_rewrite(self, state: AutoPromptState) -> str:
        """判断是否接受此次改写"""
        verification_result = state.get('verification_result', {})
        accepted = verification_result.get('accepted', False)
        
        if accepted:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 接受此次改写，继续到下一轮", level="success")
            print(f"[迭代 {state['iteration']}] [验证] 接受此次改写，继续到下一轮")
            # 清除 pending_prompt，因为已经接受了新 prompt
            state['pending_prompt'] = None
            # 清除 analysis 中的 verification_feedback，因为这是新的一轮，需要重新分析
            if state.get('analysis'):
                state['analysis'].pop('verification_feedback', None)
                state['analysis'].pop('verification_result', None)
                state['analysis'].pop('needs_rerun', None)
            return "accept"
        else:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 拒绝此次改写，回滚并重新分析", level="warning")
            print(f"[迭代 {state['iteration']}] [验证] 拒绝此次改写，回滚并重新分析")
            # 保存被拒绝的新 prompt（在回滚之前）
            rejected_prompt = state.get('current_prompt', '')
            # 回滚到改写前的 prompt
            previous_prompt = state.get('pending_prompt')
            if previous_prompt:
                state['current_prompt'] = previous_prompt
            # 将验证反馈信息添加到 analysis 中，供重新分析和改写使用
            if state.get('analysis'):
                verification_feedback = verification_result.get('feedback', '')
                verification_results = verification_result.get('verification_results', [])
                # 分析解决了和未解决的 case
                solved_cases = [r for r in verification_results if r.get('is_correct', False)]
                failed_cases = [r for r in verification_results if not r.get('is_correct', False)]
                # 更新 analysis，加入验证反馈和被拒绝的 prompt
                state['analysis']['verification_feedback'] = verification_feedback
                # 在 verification_result 中也保存被拒绝的 prompt 和解决了/未解决的 case，方便 analyze 节点使用
                verification_result['rejected_prompt'] = rejected_prompt  # 被拒绝的新 prompt
                verification_result['solved_cases'] = solved_cases  # 解决了的 case
                verification_result['failed_cases'] = failed_cases  # 未解决的 case
                state['analysis']['verification_result'] = verification_result
                state['analysis']['needs_rerun'] = True  # 标记需要重新运行
            return "reject"
    
    def _crossover_node(self, state: AutoPromptState) -> AutoPromptState:
        """杂交节点：如果有两个或以上 accepted 的候选，融合最强的两个"""
        accepted_candidates = state.get('accepted_candidates', [])
        
        # 如果只有一个或没有 accepted 的候选，跳过杂交
        if len(accepted_candidates) < 2:
            _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 只有 {len(accepted_candidates)} 个 accepted 候选，跳过杂交", level="info")
            print(f"[迭代 {state['iteration']}] [杂交] 只有 {len(accepted_candidates)} 个 accepted 候选，跳过杂交")
            return state
        
        # 获取最强的两个 accepted 候选
        prompt1_data = accepted_candidates[0]  # 最强的
        prompt2_data = accepted_candidates[1]  # 第二强的
        
        prompt1 = prompt1_data['prompt']
        prompt2 = prompt2_data['prompt']
        verification_result1 = prompt1_data['verification_result']
        verification_result2 = prompt2_data['verification_result']
        
        solve_rate1 = prompt1_data['solve_rate']
        solve_rate2 = prompt2_data['solve_rate']
        
        _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 开始融合两个最强的 accepted 候选...", level="info")
        print(f"[迭代 {state['iteration']}] [杂交] 开始融合两个最强的 accepted 候选...")
        _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 候选 1 解决率: {solve_rate1:.1%}", level="info")
        print(f"[迭代 {state['iteration']}] [杂交] 候选 1 解决率: {solve_rate1:.1%}")
        _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 候选 2 解决率: {solve_rate2:.1%}", level="info")
        print(f"[迭代 {state['iteration']}] [杂交] 候选 2 解决率: {solve_rate2:.1%}")
        
        # 获取 badcases（用于分析）
        badcases = []
        analysis = state.get('analysis', {})
        if analysis.get('has_errors', False):
            badcases = [r for r in state.get('results', []) if not r.get('is_correct', True)]
        
        # 执行杂交
        crossover_prompt = self.crossover_agent.crossover(
            prompt1=prompt1,
            prompt2=prompt2,
            verification_result1=verification_result1,
            verification_result2=verification_result2,
            badcases=badcases
        )
        
        _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 杂交完成，生成新的 prompt", level="success")
        print(f"[迭代 {state['iteration']}] [杂交] 杂交完成，生成新的 prompt")
        
        # 验证杂交后的 prompt
        if badcases:
            verification_threshold = state.get('verification_threshold', 0.5)
            _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 开始验证杂交后的 prompt...", level="info")
            print(f"[迭代 {state['iteration']}] [杂交] 开始验证杂交后的 prompt...")
            
            crossover_verification_result = self.verifier_agent.verify(
                new_prompt=crossover_prompt,
                badcases=badcases,
                threshold=verification_threshold
            )
            
            crossover_solve_rate = crossover_verification_result.get('solve_rate', 0.0)
            crossover_solved_count = crossover_verification_result.get('solved_count', 0)
            crossover_total_count = crossover_verification_result.get('total_count', 0)
            
            _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 杂交 prompt 验证结果: 解决了 {crossover_solved_count}/{crossover_total_count} 个 badcase（解决率 {crossover_solve_rate:.1%}）", level="info")
            print(f"[迭代 {state['iteration']}] [杂交] 杂交 prompt 验证结果: 解决了 {crossover_solved_count}/{crossover_total_count} 个 badcase（解决率 {crossover_solve_rate:.1%}）")
            
            # 对比杂交后的 prompt 和父代（最强的那个）
            parent_solve_rate = solve_rate1
            parent_solved_count = prompt1_data['solved_count']
            parent_total_count = prompt1_data['total_count']
            
            _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 父代（最强候选）解决率: {parent_solve_rate:.1%}", level="info")
            print(f"[迭代 {state['iteration']}] [杂交] 父代（最强候选）解决率: {parent_solve_rate:.1%}")
            
            # 选择更好的一个（解决率更高的）
            if crossover_solve_rate > parent_solve_rate:
                _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 选择杂交后的 prompt（解决率更高: {crossover_solve_rate:.1%} > {parent_solve_rate:.1%}）", level="success")
                print(f"[迭代 {state['iteration']}] [杂交] 选择杂交后的 prompt（解决率更高: {crossover_solve_rate:.1%} > {parent_solve_rate:.1%}）")
                state['current_prompt'] = crossover_prompt
                state['verification_result'] = crossover_verification_result
            elif crossover_solve_rate < parent_solve_rate:
                _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 选择父代 prompt（解决率更高: {parent_solve_rate:.1%} > {crossover_solve_rate:.1%}）", level="info")
                print(f"[迭代 {state['iteration']}] [杂交] 选择父代 prompt（解决率更高: {parent_solve_rate:.1%} > {crossover_solve_rate:.1%}）")
                # current_prompt 已经是父代（最强的那个），不需要修改
                # 但需要确保 verification_result 是父代的
                state['verification_result'] = prompt1_data['verification_result']
            else:
                # 解决率相同，选择杂交后的 prompt（尝试新版本）
                _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 解决率相同（{crossover_solve_rate:.1%}），选择杂交后的 prompt", level="info")
                print(f"[迭代 {state['iteration']}] [杂交] 解决率相同（{crossover_solve_rate:.1%}），选择杂交后的 prompt")
                state['current_prompt'] = crossover_prompt
                state['verification_result'] = crossover_verification_result
        else:
            # 没有 badcase，直接使用杂交后的 prompt
            _log_collector.log(f"[迭代 {state['iteration']}] [杂交] 没有 badcase 需要验证，直接使用杂交后的 prompt", level="info")
            print(f"[迭代 {state['iteration']}] [杂交] 没有 badcase 需要验证，直接使用杂交后的 prompt")
            state['current_prompt'] = crossover_prompt
            state['verification_result'] = {
                'accepted': True,
                'solved_count': 0,
                'total_count': 0,
                'solve_rate': 1.0,
                'verification_results': [],
                'feedback': "没有 badcase 需要验证，直接使用杂交后的 prompt。"
            }
        
        return state
    
    def _memory_node(self, state: AutoPromptState) -> AutoPromptState:
        """记忆节点：从 prompt 更改和准确率变化中总结经验"""
        _log_collector.log(f"[迭代 {state['iteration']}] [记忆] 开始总结经验...", level="info")
        print(f"[迭代 {state['iteration']}] [记忆] 开始总结经验...")
        
        current_prompt = state['current_prompt']
        current_results = state['results']  # 当前轮 predict 的结果
        
        # 从 results 计算当前准确率（不依赖 analysis，因为 memory 在 analyze 之前调用）
        current_correct = sum(1 for r in current_results if r.get('is_correct', False))
        current_total = len(current_results)
        current_acc = current_correct / current_total if current_total > 0 else 0.0
        
        # 获取上一轮的 prompt、准确率和 results
        # history 中保存的是上一轮 predict 并经过 analyze 的记录
        previous_prompt = state.get('previous_prompt', state['original_prompt'])
        prev_acc = 0.0
        prev_results = []
        
        # 从 history 中获取上一轮的数据
        if len(state['history']) >= 1:
            # history[-1] 是上一轮的记录
            prev_history_item = state['history'][-1]
            previous_prompt = prev_history_item.get('prompt', previous_prompt)
            prev_total = prev_history_item.get('total_count', 0)
            prev_errors = prev_history_item.get('error_count', 0)
            prev_acc = (prev_total - prev_errors) / prev_total if prev_total > 0 else 0.0
            prev_results = prev_history_item.get('results', [])
        
        # 只在有上一轮数据且 prompt 有变化时总结经验（第二次迭代及以后）
        if state['iteration'] > 1 and previous_prompt and previous_prompt != current_prompt and prev_results:
            experience = self.memory_agent.learn(
                old_prompt=previous_prompt,
                new_prompt=current_prompt,
                old_acc=prev_acc,
                new_acc=current_acc,
                iteration=state['iteration'],
                old_results=prev_results,
                new_results=current_results
            )
            _log_collector.log(f"[迭代 {state['iteration']}] [记忆] 经验已保存", level="success")
            print(f"[迭代 {state['iteration']}] [记忆] 经验已保存")
            # 重新加载经验以包含新的经验
            state['memory_experiences'] = self.memory_agent.load_experiences()
        else:
            _log_collector.log(f"[迭代 {state['iteration']}] [记忆] 首次迭代或prompt无变化，跳过经验总结", level="info")
            print(f"[迭代 {state['iteration']}] [记忆] 首次迭代或prompt无变化，跳过经验总结")
        
        # 更新 previous_prompt 为当前的 prompt，供下一轮使用
        state['previous_prompt'] = current_prompt
        
        # 如果还没有加载经验，现在加载
        if not state.get('memory_experiences'):
            state['memory_experiences'] = self.memory_agent.load_experiences()
        
        return state
    
    def _increment_iteration_node(self, state: AutoPromptState) -> AutoPromptState:
        """增加迭代次数节点"""
        state['iteration'] = state['iteration'] + 1
        # 清除上一轮的 verification_feedback，确保新轮次能正确记录 history
        if state.get('analysis'):
            state['analysis'].pop('verification_feedback', None)
            state['analysis'].pop('verification_result', None)
            state['analysis'].pop('needs_rerun', None)
        return state
    
    def _should_continue(self, state: AutoPromptState) -> str:
        """判断是否继续迭代"""
        # 检查是否达到最大迭代次数
        if state['iteration'] >= state['max_iterations']:
            _log_collector.log(f"[迭代 {state['iteration']}] [系统] 达到最大迭代次数 {state['max_iterations']}，对最终 prompt 进行预测验证", level="info")
            print(f"[迭代 {state['iteration']}] [系统] 达到最大迭代次数 {state['max_iterations']}，对最终 prompt 进行预测验证")
            return "final_predict"  # 最后一轮，先预测再结束
        
        # 检查是否还有错误
        if not state['analysis'].get('has_errors', True):
            _log_collector.log(f"[迭代 {state['iteration']}] [系统] 所有预测都正确，结束优化", level="success")
            print(f"[迭代 {state['iteration']}] [系统] 所有预测都正确，结束优化")
            return "end"
        
        # 继续迭代（在 predict 节点中会增加迭代次数）
        return "continue"
    
    def _final_predict_node(self, state: AutoPromptState) -> AutoPromptState:
        """最后一轮预测节点：对最终改写后的 prompt 进行预测验证"""
        _log_collector.log(f"[最终验证] [预测] 使用最终优化后的 prompt 进行预测...", level="info")
        print(f"[最终验证] [预测] 使用最终优化后的 prompt 进行预测...")
        
        results = self.prediction_agent.predict(
            state['current_prompt'],  # 使用最后一轮改写后的 prompt
            state['dataset']
        )
        
        # 打印预测结果摘要
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        _log_collector.log(f"[最终验证] [预测] 预测完成: {correct_count}/{total_count} 正确", level="success")
        print(f"[最终验证] [预测] 预测完成: {correct_count}/{total_count} 正确")
        
        # 打印每个样本的预测结果
        _log_collector.log(f"\n[最终验证] [预测] 详细预测结果:", level="info")
        print(f"\n[最终验证] [预测] 详细预测结果:")
        for i, result in enumerate(results, 1):
            input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
            prediction = result.get('prediction', '')
            ground_truth = result.get('ground_truth', '')
            is_correct = result.get('is_correct', False)
            status = "✓" if is_correct else "✗"
            level = "success" if is_correct else "error"
            log_msg = f"  {status} [{i}] 输入: {input_text}\n      预测: {prediction} | 正确答案: {ground_truth}"
            _log_collector.log(log_msg, level=level)
            print(f"  {status} [{i}] 输入: {input_text}")
            print(f"      预测: {prediction} | 正确答案: {ground_truth}")
        
        # 更新最终结果
        state['results'] = results
        
        # 记录最终验证的历史
        final_analysis = {
            'has_errors': correct_count < total_count,
            'error_count': total_count - correct_count,
            'total_count': total_count,
            'suggestions': f"最终验证：准确率 {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)"
        }
        
        # 记录最终验证的历史（只有在达到最大迭代次数时才会执行到这里）
        # 使用一个特殊的 iteration 号，避免与正常轮次重复
        final_iteration = state['iteration'] + 1
        state['history'].append({
            'iteration': final_iteration,  # 标记为最终验证轮次（通常是 max_iterations + 1）
            'prompt': state['current_prompt'],  # 最终优化后的 prompt
            'error_count': final_analysis['error_count'],
            'total_count': final_analysis['total_count'],
            'suggestions': final_analysis['suggestions'],
            'results': results,
            'is_final': True  # 标记为最终验证
        })
        _log_collector.log(f"[最终验证] [系统] 已记录最终验证历史，总 history 长度: {len(state['history'])}", level="info")
        print(f"[最终验证] [系统] 已记录最终验证历史，总 history 长度: {len(state['history'])}")
        
        state['analysis'] = final_analysis
        
        final_msg = f"\n[最终验证] [预测] 完成！最终 prompt 的准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)"
        _log_collector.log(final_msg, level="success")
        print(final_msg)
        
        return state
    
    def run(self, original_prompt: str, dataset: List[Dict[str, Any]], max_iterations: int = 3, experiment_id: str = None, verification_threshold: float = 0.5, candidate_count: int = 3) -> Dict[str, Any]:
        """
        运行优化工作流
        
        Args:
            original_prompt: 原始 prompt
            dataset: 数据集
            max_iterations: 最大迭代轮次
            experiment_id: 实验ID（如果提供，会使用新的experiment_id创建MemoryAgent）
            verification_threshold: 验证阈值（0-1），解决率 >= threshold 时接受改写，默认 0.5（50%）
            candidate_count: 遗传算法候选数量，每次改写生成 n 个版本，默认 3
            
        Returns:
            最终状态
        """
        # 如果提供了experiment_id且与当前不同，更新memory_agent
        if experiment_id and experiment_id != self.experiment_id:
            self.experiment_id = experiment_id
            self.memory_agent = MemoryAgent(experiment_id=experiment_id)
        initial_state: AutoPromptState = {
            'original_prompt': original_prompt,
            'current_prompt': original_prompt,
            'dataset': dataset,
            'results': [],
            'analysis': {},
            'iteration': 1,
            'max_iterations': max_iterations,
            'history': [],
            'previous_prompt': original_prompt,
            'memory_experiences': self.memory_agent.load_experiences(),
            'verification_result': {},
            'pending_prompt': None,
            'verification_threshold': verification_threshold,
            'candidate_prompts': [],
            'candidate_count': candidate_count,
            'accepted_candidates': []
        }
        
        # 清空之前的日志
        _log_collector.clear()
        
        # 设置递归限制配置，确保有足够的迭代空间
        # 每次迭代大约需要 4-5 个节点（predict, memory, analyze, rewrite, increment_iteration）
        # 加上 final_predict，最多需要 (max_iterations * 5 + 1) 次递归
        recursion_limit = max(50, max_iterations * 10)  # 至少50次，或根据迭代次数动态计算
        config = {"recursion_limit": recursion_limit}
        
        _log_collector.log("=" * 60, level="info")
        _log_collector.log(f"[系统] 开始优化任务 - 最大迭代次数: {max_iterations}", level="info")
        _log_collector.log("=" * 60, level="info")
        
        final_state = self.graph.invoke(initial_state, config=config)
        
        # 打印 history 信息用于调试
        _log_collector.log(f"[系统] 优化完成，总 history 记录数: {len(final_state['history'])}", level="info")
        for i, h in enumerate(final_state['history']):
            acc = ((h.get('total_count', 0) - h.get('error_count', 0)) / h.get('total_count', 1) * 100) if h.get('total_count', 0) > 0 else 0
            _log_collector.log(f"[系统] History[{i}]: 迭代 {h.get('iteration', 'N/A')}, 准确率: {acc:.1f}%, is_final: {h.get('is_final', False)}", level="info")
        print(f"[系统] 优化完成，总 history 记录数: {len(final_state['history'])}")
        for i, h in enumerate(final_state['history']):
            acc = ((h.get('total_count', 0) - h.get('error_count', 0)) / h.get('total_count', 1) * 100) if h.get('total_count', 0) > 0 else 0
            print(f"  History[{i}]: 迭代 {h.get('iteration', 'N/A')}, 准确率: {acc:.1f}%, is_final: {h.get('is_final', False)}")
        
        # 计算最终准确率
        final_results = final_state.get('results', [])
        final_correct_count = sum(1 for r in final_results if r.get('is_correct', False))
        final_total_count = len(final_results)
        final_accuracy = (final_correct_count / final_total_count * 100) if final_total_count > 0 else 0.0
        
        # 查找历史最高准确率的记录（排除最终验证记录）
        best_history = None
        best_accuracy = 0.0
        best_iteration = 0
        
        for h in final_state['history']:
            if h.get('is_final', False):
                continue  # 跳过最终验证记录
            
            acc = ((h.get('total_count', 0) - h.get('error_count', 0)) / h.get('total_count', 1) * 100) if h.get('total_count', 0) > 0 else 0
            if acc > best_accuracy:
                best_accuracy = acc
                best_history = h
                best_iteration = h.get('iteration', 0)
        
        # 如果最终准确率未达到100%，且历史最高准确率更高，使用历史最高准确率的 prompt
        final_prompt = final_state['current_prompt']
        prompt_source = "最终轮次"
        prompt_iteration = final_state['iteration']
        
        if final_accuracy < 100.0 and best_history and best_accuracy > final_accuracy:
            final_prompt = best_history.get('prompt', final_state['current_prompt'])
            prompt_source = "历史最高准确率"
            prompt_iteration = best_iteration
            _log_collector.log(f"[系统] 最终准确率 {final_accuracy:.1f}% 未达到 100%，使用第 {best_iteration} 轮的 prompt（准确率 {best_accuracy:.1f}%）", level="warning")
            print(f"[系统] 最终准确率 {final_accuracy:.1f}% 未达到 100%，使用第 {best_iteration} 轮的 prompt（准确率 {best_accuracy:.1f}%）")
        elif final_accuracy < 100.0 and best_history and best_accuracy == final_accuracy:
            # 如果最终准确率等于历史最高，但未达到100%，也使用历史最高
            final_prompt = best_history.get('prompt', final_state['current_prompt'])
            prompt_source = "历史最高准确率"
            prompt_iteration = best_iteration
            _log_collector.log(f"[系统] 最终准确率 {final_accuracy:.1f}% 未达到 100%，使用第 {best_iteration} 轮的 prompt（准确率 {best_accuracy:.1f}%）", level="warning")
            print(f"[系统] 最终准确率 {final_accuracy:.1f}% 未达到 100%，使用第 {best_iteration} 轮的 prompt（准确率 {best_accuracy:.1f}%）")
        elif final_accuracy >= 100.0:
            _log_collector.log(f"[系统] 最终准确率 {final_accuracy:.1f}% 达到 100%，使用最终轮次的 prompt", level="success")
            print(f"[系统] 最终准确率 {final_accuracy:.1f}% 达到 100%，使用最终轮次的 prompt")
        
        return {
            'final_prompt': final_prompt,
            'final_prompt_source': prompt_source,  # 标记 prompt 来源：最终轮次 / 历史最高准确率
            'final_prompt_iteration': prompt_iteration,  # prompt 对应的轮次
            'final_results': final_state['results'],
            'final_analysis': final_state['analysis'],
            'final_accuracy': final_accuracy,  # 最终准确率
            'best_accuracy': best_accuracy if best_history else final_accuracy,  # 历史最高准确率
            'best_iteration': best_iteration if best_history else final_state['iteration'],  # 历史最高准确率对应的轮次
            'history': final_state['history'],
            'iterations': final_state['iteration'],
            'all_iteration_results': final_state['history'],  # 包含所有轮次的详细结果
            'memory_experiences': final_state.get('memory_experiences', '')  # 包含累积的经验
        }

