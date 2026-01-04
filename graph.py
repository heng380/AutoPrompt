"""
使用 LangGraph 构建的 AutoPrompt 优化工作流
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from agents import PredictionAgent, AnalysisAgent, RewriteAgent, MemoryAgent


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


class AutoPromptGraph:
    """AutoPrompt 优化工作流图"""
    
    def __init__(self, experiment_id: str = None):
        self.experiment_id = experiment_id
        self.prediction_agent = PredictionAgent()
        self.analysis_agent = AnalysisAgent()
        self.rewrite_agent = RewriteAgent()
        self.memory_agent = MemoryAgent(experiment_id=experiment_id)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AutoPromptState)
        
        # 添加节点
        workflow.add_node("predict", self._predict_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("memory", self._memory_node)
        workflow.add_node("final_predict", self._final_predict_node)  # 最后一轮预测节点
        
        # 设置入口点
        workflow.add_edge(START, "predict")
        
        # 添加边
        workflow.add_edge("predict", "analyze")
        workflow.add_edge("analyze", "rewrite")
        workflow.add_edge("rewrite", "memory")
        
        # 添加一个更新迭代次数的节点
        workflow.add_node("increment_iteration", self._increment_iteration_node)
        
        # 条件边：决定是否继续迭代（从memory节点出发）
        workflow.add_conditional_edges(
            "memory",
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
        print(f"[迭代 {state['iteration']}] 开始预测...")
        
        results = self.prediction_agent.predict(
            state['current_prompt'],
            state['dataset']
        )
        
        # 打印预测结果摘要
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        print(f"[迭代 {state['iteration']}] 预测完成: {correct_count}/{total_count} 正确")
        
        # 打印每个样本的预测结果
        print(f"\n[迭代 {state['iteration']}] 详细预测结果:")
        for i, result in enumerate(results, 1):
            input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
            prediction = result.get('prediction', '')
            ground_truth = result.get('ground_truth', '')
            is_correct = result.get('is_correct', False)
            status = "✓" if is_correct else "✗"
            print(f"  {status} [{i}] 输入: {input_text}")
            print(f"      预测: {prediction} | 正确答案: {ground_truth}")
        
        state['results'] = results
        return state
    
    def _analyze_node(self, state: AutoPromptState) -> AutoPromptState:
        """分析节点：分析错误案例"""
        print(f"[迭代 {state['iteration']}] 开始分析错误案例...")
        
        analysis = self.analysis_agent.analyze(
            state['current_prompt'],
            state['results']
        )
        
        state['analysis'] = analysis
        
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
    
    def _rewrite_node(self, state: AutoPromptState) -> AutoPromptState:
        """改写节点：根据分析结果改写 prompt"""
        print(f"[迭代 {state['iteration']}] 开始改写 prompt...")
        
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
        
        state['current_prompt'] = new_prompt
        return state
    
    def _memory_node(self, state: AutoPromptState) -> AutoPromptState:
        """记忆节点：从 prompt 更改和准确率变化中总结经验"""
        print(f"[迭代 {state['iteration']}] 开始总结经验...")
        
        current_prompt = state['current_prompt']
        current_results = state['results']  # 当前轮的预测结果
        
        # 计算当前准确率
        current_analysis = state['analysis']
        current_total = current_analysis.get('total_count', 0)
        current_errors = current_analysis.get('error_count', 0)
        current_acc = (current_total - current_errors) / current_total if current_total > 0 else 0.0
        
        # 获取上一轮的 prompt、准确率和results
        # history[-1] 是当前iteration的记录（刚刚由analyze节点添加）
        # history[-2] 是上一次iteration的记录（如果存在）
        previous_prompt = state.get('previous_prompt', state['original_prompt'])
        prev_acc = 0.0
        prev_results = []
        
        if len(state['history']) >= 2:
            # 有上一轮的数据，从history[-2]获取
            prev_history_item = state['history'][-2]
            previous_prompt = prev_history_item.get('prompt', previous_prompt)
            prev_total = prev_history_item.get('total_count', 0)
            prev_errors = prev_history_item.get('error_count', 0)
            prev_acc = (prev_total - prev_errors) / prev_total if prev_total > 0 else 0.0
            prev_results = prev_history_item.get('results', [])
        elif state['iteration'] > 1:
            # 第二次迭代，但history中只有一条记录（当前iteration的），使用previous_prompt
            # previous_prompt应该已经在第一次迭代后更新了
            # 对于results，需要从第一次迭代的history中获取
            if len(state['history']) >= 1:
                first_history_item = state['history'][0]
                prev_results = first_history_item.get('results', [])
        
        # 只在有变化时总结经验（第二次迭代及以后，且prompt有变化）
        if state['iteration'] > 1 and previous_prompt != current_prompt:
            experience = self.memory_agent.learn(
                old_prompt=previous_prompt,
                new_prompt=current_prompt,
                old_acc=prev_acc,
                new_acc=current_acc,
                iteration=state['iteration'],
                old_results=prev_results,
                new_results=current_results
            )
            print(f"[迭代 {state['iteration']}] 经验已保存")
            # 重新加载经验以包含新的经验
            state['memory_experiences'] = self.memory_agent.load_experiences()
        else:
            print(f"[迭代 {state['iteration']}] 首次迭代或prompt无变化，跳过经验总结")
        
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
            print(f"达到最大迭代次数 {state['max_iterations']}，对最终 prompt 进行预测验证")
            return "final_predict"  # 最后一轮，先预测再结束
        
        # 检查是否还有错误
        if not state['analysis'].get('has_errors', True):
            print("所有预测都正确，结束优化")
            return "end"
        
        # 继续迭代（在 predict 节点中会增加迭代次数）
        return "continue"
    
    def _final_predict_node(self, state: AutoPromptState) -> AutoPromptState:
        """最后一轮预测节点：对最终改写后的 prompt 进行预测验证"""
        print(f"[最终验证] 使用最终优化后的 prompt 进行预测...")
        
        results = self.prediction_agent.predict(
            state['current_prompt'],  # 使用最后一轮改写后的 prompt
            state['dataset']
        )
        
        # 打印预测结果摘要
        correct_count = sum(1 for r in results if r.get('is_correct', False))
        total_count = len(results)
        print(f"[最终验证] 预测完成: {correct_count}/{total_count} 正确")
        
        # 打印每个样本的预测结果
        print(f"\n[最终验证] 详细预测结果:")
        for i, result in enumerate(results, 1):
            input_text = result.get('input', '')[:50] + '...' if len(str(result.get('input', ''))) > 50 else result.get('input', '')
            prediction = result.get('prediction', '')
            ground_truth = result.get('ground_truth', '')
            is_correct = result.get('is_correct', False)
            status = "✓" if is_correct else "✗"
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
        
        print(f"\n[最终验证] 完成！最终 prompt 的准确率: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)")
        
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
            'memory_experiences': self.memory_agent.load_experiences()
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            'final_prompt': final_state['current_prompt'],
            'final_results': final_state['results'],
            'final_analysis': final_state['analysis'],
            'history': final_state['history'],
            'iterations': final_state['iteration'],
            'all_iteration_results': final_state['history'],  # 包含所有轮次的详细结果
            'memory_experiences': final_state.get('memory_experiences', '')  # 包含累积的经验
        }

