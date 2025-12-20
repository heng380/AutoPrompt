# 工作流程说明

## Prompt 流转逻辑

### 工作流程

假设进行 5 轮迭代，流程如下：

```
初始状态：
- iteration = 1
- current_prompt = 原始 prompt

第 1 轮：
1. predict: 使用 current_prompt（原始 prompt）进行预测
2. analyze: 分析错误案例
3. 记录历史: 保存第1轮使用的 prompt（原始 prompt）和预测结果
4. rewrite: 根据分析结果改写 prompt → 更新 current_prompt（第1轮改写后的 prompt）
5. 判断是否继续 → 继续

第 2 轮：
1. increment_iteration: iteration = 2
2. predict: 使用 current_prompt（第1轮改写后的 prompt）进行预测
3. analyze: 分析错误案例
4. 记录历史: 保存第2轮使用的 prompt（第1轮改写后的 prompt）和预测结果
5. rewrite: 根据分析结果改写 prompt → 更新 current_prompt（第2轮改写后的 prompt）
6. 判断是否继续 → 继续

... 如此循环 ...

第 5 轮：
1. increment_iteration: iteration = 5
2. predict: 使用 current_prompt（第4轮改写后的 prompt）进行预测
3. analyze: 分析错误案例
4. 记录历史: 保存第5轮使用的 prompt（第4轮改写后的 prompt）和预测结果
5. rewrite: 根据分析结果改写 prompt → 更新 current_prompt（第5轮改写后的 prompt）
6. 判断是否继续 → 达到最大迭代次数，结束

最终：
- final_prompt = current_prompt（第5轮改写后的 prompt）
```

### 关键点

1. **历史记录中的 prompt**：
   - 第 N 轮历史中保存的 prompt 是**第 N 轮使用的 prompt**
   - 第 N 轮使用的 prompt = 第 N-1 轮改写后得到的（除了第1轮使用原始 prompt）

2. **final_prompt**：
   - 是最后一轮（第5轮）改写后得到的 prompt
   - 但这个 prompt **还没有被用来进行预测**
   - 它是在第5轮预测和分析之后，根据第5轮的错误案例改写得到的

3. **第5轮历史中的 prompt**：
   - 是第4轮改写后得到的 prompt
   - 是第5轮用来进行预测的 prompt

### 示例

假设进行了 5 轮迭代：

- **原始 prompt**: "你是一个分类助手"
- **第1轮历史中的 prompt**: "你是一个分类助手"（原始 prompt）
- **第2轮历史中的 prompt**: "你是一个分类助手。请仔细分析输入。"（第1轮改写后）
- **第3轮历史中的 prompt**: "你是一个分类助手。请仔细分析输入。只返回数字。"（第2轮改写后）
- **第4轮历史中的 prompt**: "你是一个分类助手。请仔细分析输入。只返回数字。不要包含其他文字。"（第3轮改写后）
- **第5轮历史中的 prompt**: "你是一个分类助手。请仔细分析输入。只返回数字。不要包含其他文字。确保答案准确。"（第4轮改写后）
- **final_prompt**: "你是一个分类助手。请仔细分析输入。只返回数字。不要包含其他文字。确保答案准确。考虑上下文信息。"（第5轮改写后，但未用于预测）

### 总结

- **第5轮历史中的 prompt** ≠ **final_prompt**
- **第5轮历史中的 prompt** = 第4轮改写后得到的，用于第5轮预测
- **final_prompt** = 第5轮改写后得到的，是最新的优化版本，但还没有经过预测验证

