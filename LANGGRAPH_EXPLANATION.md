# LangGraph 工作流机制详解

## 1. LangGraph 核心概念

### 1.1 什么是 LangGraph？
你是一个文本分类专家。你的任务是根据新闻标题headline预测其 Typography 类别是否为标题党

输入：一个新闻标题
输出：0 或 1（只返回数字，不要包含其他文字, 1 为标题党）

请仔细分析标题的内容、风格和特征，然后给出分类结果。◊
LangGraph 是 LangChain 的一个扩展，用于构建有状态的、多步骤的工作流（workflow）。它使用图（Graph）的结构来编排多个 Agent 的执行顺序。

### 1.2 核心组件

- **State（状态）**: 工作流中所有节点共享的数据结构，使用 TypedDict 定义
- **Node（节点）**: 工作流中的一个执行单元，通常对应一个 Agent 或处理函数
- **Edge（边）**: 节点之间的连接，定义执行流程
- **Conditional Edge（条件边）**: 根据状态动态决定下一个节点

## 2. 本项目的工作流编排

### 2.1 状态定义（AutoPromptState）

```python
class AutoPromptState(TypedDict, total=False):
    original_prompt: str          # 原始 prompt（不变）
    current_prompt: str           # 当前轮次的 prompt（会被更新）
    dataset: List[Dict]           # 数据集（不变）
    results: List[Dict]           # 预测结果（每轮更新）
    analysis: Dict                # 分析结果（每轮更新）
    iteration: int                # 当前迭代轮次
    max_iterations: int           # 最大迭代次数
    history: List[Dict]           # 历史记录（累积）
    previous_prompt: str          # 上一轮的 prompt
    memory_experiences: str       # 累积的经验文本
```

**关键点**：
- `total=False` 表示所有字段都是可选的
- 状态在整个工作流中传递和更新
- 每个节点可以读取和修改状态

### 2.2 工作流图结构

```
START
  ↓
predict (预测节点)
  ↓
analyze (分析节点)
  ↓
rewrite (改写节点)
  ↓
memory (记忆节点)
  ↓
  ├─→ [条件判断: _should_continue]
  │     ├─→ "continue" → increment_iteration → predict (继续迭代)
  │     ├─→ "final_predict" → final_predict → END (最后一轮验证)
  │     └─→ "end" → END (所有正确，提前结束)
```

### 2.3 节点详解

#### 节点1: predict（预测节点）

**功能**: 使用当前 prompt 对数据集进行预测

**输入**（从 state 读取）:
- `current_prompt`: 当前的 prompt
- `dataset`: 数据集

**处理**:
```python
results = prediction_agent.predict(current_prompt, dataset)
# 每个样本返回: {input, prediction, ground_truth, is_correct}
```

**输出**（更新 state）:
- `results`: 预测结果列表

**传递给下一个节点**: 完整的 state（包含 results）

---

#### 节点2: analyze（分析节点）

**功能**: 分析预测结果中的错误案例，生成改进建议

**输入**（从 state 读取）:
- `current_prompt`: 当前的 prompt
- `results`: 预测结果

**处理**:
```python
analysis = analysis_agent.analyze(current_prompt, results)
# 返回: {has_errors, error_count, total_count, suggestions, error_cases}
```

**输出**（更新 state）:
- `analysis`: 分析结果字典
- `history`: 追加当前轮次的记录

**传递给下一个节点**: 完整的 state（包含 analysis 和更新的 history）

---

#### 节点3: rewrite（改写节点）

**功能**: 根据分析结果和历史经验改写 prompt

**输入**（从 state 读取）:
- `original_prompt`: 原始 prompt（作为参考）
- `analysis`: 分析结果
- `memory_experiences`: 历史经验（从文件加载或从 state 读取）
- `iteration`: 当前迭代轮次

**处理**:
```python
new_prompt = rewrite_agent.rewrite(
    original_prompt, 
    analysis, 
    iteration, 
    memory_experiences
)
```

**输出**（更新 state）:
- `current_prompt`: 新的 prompt

**传递给下一个节点**: 完整的 state（包含更新后的 current_prompt）

---

#### 节点4: memory（记忆节点）

**功能**: 从 prompt 更改和准确率变化中总结经验

**输入**（从 state 读取）:
- `current_prompt`: 当前轮次的 prompt
- `results`: 当前轮的预测结果
- `history`: 历史记录（用于获取上一轮的数据）
- `previous_prompt`: 上一轮的 prompt

**处理**:
```python
# 比较上一轮和当前轮
experience = memory_agent.learn(
    old_prompt=previous_prompt,
    new_prompt=current_prompt,
    old_acc=prev_acc,
    new_acc=current_acc,
    iteration=iteration,
    old_results=prev_results,
    new_results=current_results
)
# 经验保存到文件 memory_experiences.txt
```

**输出**（更新 state）:
- `memory_experiences`: 累积的经验文本
- `previous_prompt`: 更新为当前 prompt（供下一轮使用）

**传递给下一个节点**: 完整的 state（用于条件判断）

---

#### 节点5: increment_iteration（迭代递增节点）

**功能**: 增加迭代轮次计数

**输入**（从 state 读取）:
- `iteration`: 当前迭代轮次

**输出**（更新 state）:
- `iteration`: iteration + 1

**传递给下一个节点**: 完整的 state（回到 predict 节点继续迭代）

---

#### 节点6: final_predict（最终预测节点）

**功能**: 对最终优化后的 prompt 进行最后一次预测验证

**输入**（从 state 读取）:
- `current_prompt`: 最终优化后的 prompt
- `dataset`: 数据集

**处理**:
```python
results = prediction_agent.predict(current_prompt, dataset)
```

**输出**（更新 state）:
- `results`: 最终预测结果
- `analysis`: 最终分析结果
- `history`: 追加最终验证的记录

**传递给下一个节点**: END（工作流结束）

---

### 2.4 条件边（Conditional Edge）

**位置**: memory 节点之后

**判断函数**: `_should_continue(state)`

**逻辑**:
```python
def _should_continue(state):
    # 1. 达到最大迭代次数 → "final_predict"
    if state['iteration'] >= state['max_iterations']:
        return "final_predict"
    
    # 2. 所有预测都正确 → "end"
    if not state['analysis'].get('has_errors', True):
        return "end"
    
    # 3. 继续迭代 → "continue"
    return "continue"
```

**路由**:
- `"continue"` → `increment_iteration` → `predict`（继续迭代）
- `"final_predict"` → `final_predict` → END（最后一轮验证）
- `"end"` → END（提前结束）

## 3. 数据传递机制

### 3.1 状态传递流程

每个节点函数都遵循以下模式：

```python
def _node_function(self, state: AutoPromptState) -> AutoPromptState:
    # 1. 从 state 读取需要的数据
    data = state['field_name']
    
    # 2. 调用相应的 agent 处理
    result = self.agent.method(data)
    
    # 3. 更新 state
    state['field_name'] = result
    
    # 4. 返回更新后的 state
    return state
```

### 3.2 各节点之间的数据传递

#### predict → analyze
```
state = {
    'current_prompt': "当前的 prompt",
    'dataset': [...],
    'results': [  # ← predict 节点添加
        {
            'input': "输入文本",
            'prediction': "预测结果",
            'ground_truth': "正确答案",
            'is_correct': True/False
        },
        ...
    ],
    ...
}
```

#### analyze → rewrite
```
state = {
    'results': [...],
    'analysis': {  # ← analyze 节点添加
        'has_errors': True,
        'error_count': 3,
        'total_count': 10,
        'suggestions': "改进建议文本...",
        'error_cases': [...]
    },
    'history': [  # ← analyze 节点追加
        {
            'iteration': 1,
            'prompt': "...",
            'error_count': 3,
            'total_count': 10,
            'suggestions': "...",
            'results': [...]
        }
    ],
    ...
}
```

#### rewrite → memory
```
state = {
    'analysis': {...},
    'current_prompt': "新的 prompt",  # ← rewrite 节点更新
    'memory_experiences': "历史经验文本...",
    ...
}
```

#### memory → (条件判断) → increment_iteration
```
state = {
    'current_prompt': "新的 prompt",
    'previous_prompt': "新的 prompt",  # ← memory 节点更新（供下一轮使用）
    'memory_experiences': "更新后的经验文本...",  # ← memory 节点更新
    'results': [...],
    'analysis': {...},
    ...
}
```

## 4. 执行流程示例

### 第一次迭代

1. **START** → 初始化 state
2. **predict** → 使用原始 prompt 预测
3. **analyze** → 分析错误，生成建议
4. **rewrite** → 根据建议改写 prompt（参考历史经验）
5. **memory** → 跳过（第一次迭代，没有上一轮数据）
6. **条件判断** → 返回 "continue"
7. **increment_iteration** → iteration: 1 → 2

### 第二次迭代

1. **predict** → 使用新的 prompt 预测
2. **analyze** → 分析错误，生成建议
3. **rewrite** → 根据建议和历史经验改写 prompt
4. **memory** → 比较第一轮和第二轮的差异，总结经验
5. **条件判断** → 返回 "continue"
6. **increment_iteration** → iteration: 2 → 3

### 最后一次迭代（达到 max_iterations）

1. **predict** → 预测
2. **analyze** → 分析
3. **rewrite** → 改写
4. **memory** → 总结经验
5. **条件判断** → 返回 "final_predict"（达到最大迭代次数）
6. **final_predict** → 最终验证
7. **END** → 返回最终结果

## 5. 关键设计模式

### 5.1 状态共享

- 所有节点共享同一个 state 对象
- 节点通过读取和修改 state 来传递数据
- 状态在节点之间自动传递

### 5.2 函数式更新

- 每个节点函数接收 state，返回更新后的 state
- LangGraph 自动合并状态更新
- 支持部分更新（TypedDict with total=False）

### 5.3 条件路由

- 使用条件边实现动态流程控制
- 根据状态值决定下一步执行路径
- 支持循环和提前终止

### 5.4 Agent 封装

- 每个 Agent 封装特定的功能
- Agent 之间通过 state 间接通信
- 保持 Agent 的独立性和可测试性

## 6. 总结

LangGraph 的核心优势：

1. **状态管理**: 自动管理复杂的状态传递
2. **可视化流程**: 图结构清晰表达工作流逻辑
3. **灵活控制**: 条件边支持动态路由
4. **模块化**: 节点和 Agent 分离，易于维护和扩展
5. **类型安全**: TypedDict 提供类型提示和验证

在这个项目中，LangGraph 实现了：
- 多轮迭代的 prompt 优化流程
- 多个 Agent 的协调工作
- 状态的累积和传递（history, memory_experiences）
- 动态的流程控制（条件判断继续/结束）

