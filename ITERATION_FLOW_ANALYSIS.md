# 迭代流程分析（以5轮为例）

## 完整流程追踪

假设 `max_iterations = 5`

### 第1轮（iteration=1）

1. **predict(原始prompt)** → results_1, acc_1
2. **memory** → 跳过（首次迭代，没有上一轮数据）
3. **analyze** → 记录到 history: `{iteration:1, prompt:原始prompt, results:results_1}`
4. **rewrite** → 生成 prompt_2
5. **判断** → iteration=1 < 5，返回 "continue"
6. **increment_iteration** → iteration=2

### 第2轮（iteration=2）

1. **predict(prompt_2)** → results_2, acc_2
2. **memory** → 对比：原始prompt vs prompt_2, acc_1 vs acc_2
3. **analyze** → 记录到 history: `{iteration:2, prompt:prompt_2, results:results_2}`
4. **rewrite** → 生成 prompt_3
5. **判断** → iteration=2 < 5，返回 "continue"
6. **increment_iteration** → iteration=3

### 第3轮（iteration=3）

1. **predict(prompt_3)** → results_3, acc_3
2. **memory** → 对比：prompt_2 vs prompt_3, acc_2 vs acc_3
3. **analyze** → 记录到 history
4. **rewrite** → 生成 prompt_4
5. **判断** → iteration=3 < 5，返回 "continue"
6. **increment_iteration** → iteration=4

### 第4轮（iteration=4）

1. **predict(prompt_4)** → results_4, acc_4
2. **memory** → 对比：prompt_3 vs prompt_4, acc_3 vs acc_4
3. **analyze** → 记录到 history
4. **rewrite** → 生成 prompt_5
5. **判断** → iteration=4 < 5，返回 "continue"
6. **increment_iteration** → iteration=5

### 第5轮（iteration=5）

1. **predict(prompt_5)** → results_5, acc_5
2. **memory** → 对比：prompt_4 vs prompt_5, acc_4 vs acc_5
3. **analyze** → 记录到 history
4. **rewrite** → 生成 prompt_6（最终优化后的prompt）
5. **判断** → iteration=5 >= 5，返回 **"final_predict"**
6. **final_predict(prompt_6)** → 验证最终prompt的效果
7. **END**

## 关键点

### 第五轮后的 final_predict

**是的，第五轮优化完后还会进行一次 final_predict！**

原因：
- 第5轮的 `predict` 验证的是 `prompt_5`
- 第5轮的 `rewrite` 生成了 `prompt_6`（最终优化后的prompt）
- 但 `prompt_6` 还没有经过验证
- 所以 `final_predict` 节点会用 `prompt_6` 进行一次预测验证

### final_predict 的作用

`final_predict` 节点会：
1. 使用最后一轮 rewrite 后的 prompt（prompt_6）进行预测
2. 记录最终验证结果到 history
3. 更新 state 中的 results 和 analysis
4. 确保返回的结果是基于最终优化后的 prompt

### 总结

- **第5轮的 predict**：验证 prompt_5 的效果
- **第5轮的 rewrite**：生成 prompt_6
- **final_predict**：验证 prompt_6 的效果（这是最终返回的结果）

所以最终返回的 `final_results` 是基于 `prompt_6` 的预测结果，而不是 `prompt_5` 的结果。

