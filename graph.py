"""
使用 LangGraph 构建的 AutoPrompt 优化工作流
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agents import PredictionAgent, AnalysisAgent, RewriteAgent, MemoryAgent, VerifierAgent
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


class AutoPromptGraph:
    """AutoPrompt 优化工作流图"""
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id
        self.prediction_agent = PredictionAgent()
        self.analysis_agent = AnalysisAgent()
        self.rewrite_agent = RewriteAgent()
        self.memory_agent = MemoryAgent(experiment_id=experiment_id)
        self.verifier_agent = VerifierAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AutoPromptState)
        
        # 添加节点
        workflow.add_node("predict", self._predict_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("verify", self._verify_node)  # 验证节点：验证改写后的 prompt
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
                "accept": "check_continue",  # 接受改写，检查是否继续
                "reject": "analyze"  # 拒绝改写，回滚后重新分析
            }
        )
        
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
            # 在建议中加入验证反馈信息
            analysis['suggestions'] = (
                f"【验证反馈】{saved_verification_feedback}\n\n"
                f"原改进建议：\n{analysis.get('suggestions', '')}\n\n"
                f"请结合验证反馈，重新思考如何改进 prompt，确保能够解决至少一半的 badcase。"
            )
        
        state['analysis'] = analysis
        
        # 只在首次分析时记录历史（验证失败后重新分析不重复记录）
        if not has_verification_feedback:
            # 记录历史（包含每轮的详细预测结果）
            state['history'].append({
                'iteration': state['iteration'],
                'prompt': state['current_prompt'],
                'error_count': analysis.get('error_count', 0),
                'total_count': analysis.get('total_count', 0),
                'suggestions': analysis.get('suggestions', ''),
                'results': state['results']  # 保存每轮的详细预测结果
            })
        
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
        """改写节点：根据分析结果改写 prompt"""
        _log_collector.log(f"[迭代 {state['iteration']}] [改写] 开始改写 prompt...", level="info")
        print(f"[迭代 {state['iteration']}] [改写] 开始改写 prompt...")
        
        # 保存当前 prompt 作为待验证的 prompt（如果验证失败可以回滚）
        state['pending_prompt'] = state['current_prompt']
        
        # 获取已有的经验
        memory_experiences = state.get('memory_experiences', '')
        if not memory_experiences:
            memory_experiences = self.memory_agent.load_experiences()
            state['memory_experiences'] = memory_experiences
        
        new_prompt = self.rewrite_agent.rewrite(
            state['original_prompt'],
            state['analysis'],
            state['iteration'],
            memory_experiences
        )
        
        # 保存改写后的 prompt（待验证）
        state['current_prompt'] = new_prompt
        return state
    
    def _verify_node(self, state: AutoPromptState) -> AutoPromptState:
        """验证节点：验证改写后的 prompt 是否能解决目标 badcase"""
        _log_collector.log(f"[迭代 {state['iteration']}] [验证] 开始验证改写后的 prompt...", level="info")
        print(f"[迭代 {state['iteration']}] [验证] 开始验证改写后的 prompt...")
        
        # 获取当前轮次的 badcase（错误案例）
        # 这些 badcase 是改写要解决的目标
        # 需要获取所有的 badcase，而不仅仅是 analysis 中返回的样本
        badcases = []
        analysis = state.get('analysis', {})
        if analysis.get('has_errors', False):
            # 从当前轮次的 results 中筛选所有错误案例（完整的 badcase）
            badcases = [r for r in state.get('results', []) if not r.get('is_correct', True)]
        
        if not badcases:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 没有 badcase 需要验证，自动接受改写", level="info")
            print(f"[迭代 {state['iteration']}] [验证] 没有 badcase 需要验证，自动接受改写")
            state['verification_result'] = {
                'accepted': True,
                'solved_count': 0,
                'total_count': 0,
                'solve_rate': 1.0,
                'verification_results': [],
                'feedback': "没有 badcase 需要验证，自动接受改写。"
            }
            return state
        
        _log_collector.log(f"[迭代 {state['iteration']}] [验证] 需要验证 {len(badcases)} 个 badcase...", level="info")
        print(f"[迭代 {state['iteration']}] [验证] 需要验证 {len(badcases)} 个 badcase...")
        
        # 使用改写后的 prompt 验证 badcase
        verification_result = self.verifier_agent.verify(
            new_prompt=state['current_prompt'],
            badcases=badcases
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
            return "accept"
        else:
            _log_collector.log(f"[迭代 {state['iteration']}] [验证] 拒绝此次改写，回滚并重新分析", level="warning")
            print(f"[迭代 {state['iteration']}] [验证] 拒绝此次改写，回滚并重新分析")
            # 回滚到改写前的 prompt
            previous_prompt = state.get('pending_prompt')
            if previous_prompt:
                state['current_prompt'] = previous_prompt
            # 将验证反馈信息添加到 analysis 中，供重新分析和改写使用
            if state.get('analysis'):
                verification_feedback = verification_result.get('feedback', '')
                # 更新 analysis，加入验证反馈
                state['analysis']['verification_feedback'] = verification_feedback
                state['analysis']['verification_result'] = verification_result
                state['analysis']['needs_rerun'] = True  # 标记需要重新运行
            return "reject"
    
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
        
        state['history'].append({
            'iteration': state['iteration'] + 1,  # 标记为最终验证轮次
            'prompt': state['current_prompt'],  # 最终优化后的 prompt
            'error_count': final_analysis['error_count'],
            'total_count': final_analysis['total_count'],
            'suggestions': final_analysis['suggestions'],
            'results': results,
            'is_final': True  # 标记为最终验证
        })
        
        state['analysis'] = final_analysis
        
        final_msg = f"\n[最终验证] [预测] 完成！最终 prompt 的准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)"
        _log_collector.log(final_msg, level="success")
        print(final_msg)
        
        return state
    
    def run(self, original_prompt: str, dataset: List[Dict[str, Any]], max_iterations: int = 5, experiment_id: str = None) -> Dict[str, Any]:
        """
        运行优化工作流
        
        Args:
            original_prompt: 原始 prompt
            dataset: 数据集
            max_iterations: 最大迭代轮次
            experiment_id: 实验ID（如果提供，会使用新的experiment_id创建MemoryAgent）
            
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
            'pending_prompt': None
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
        
        return {
            'final_prompt': final_state['current_prompt'],
            'final_results': final_state['results'],
            'final_analysis': final_state['analysis'],
            'history': final_state['history'],
            'iterations': final_state['iteration'],
            'all_iteration_results': final_state['history'],  # 包含所有轮次的详细结果
            'memory_experiences': final_state.get('memory_experiences', '')  # 包含累积的经验
        }

