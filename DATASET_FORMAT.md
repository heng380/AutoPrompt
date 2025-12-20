# 数据集格式说明

## 概述

本项目对数据集的列名**没有严格要求**，系统会自动识别常见的列名。最重要的是在 **prompt 中说明**如何使用数据。

## 自动识别的列名

### 输入列（Input Column）

系统会自动识别以下列名作为输入数据（不区分大小写）：

- `input`
- `question`
- `text`
- `query`
- `prompt`
- `headline`
- `title`
- `content`

### 标准答案列（Ground Truth Column）

系统会自动识别以下列名作为标准答案（不区分大小写）：

- `ground_truth`
- `answer`
- `label`
- `output`
- `target`
- `typography`
- `category`
- `class`

## 如果列名不在列表中

如果数据集的列名不在上述列表中，系统会：

1. **输入列**：自动使用第二列作为输入（假设第一列是 ID）
2. **标准答案列**：如果找不到，`ground_truth` 会为空（系统仍会运行，但无法评估准确率）

## 重要：在 Prompt 中说明

**最关键的是在 prompt 中说明如何使用数据**，因为：

1. 系统会将整个数据项传递给模型
2. Prompt 会告诉模型如何处理输入数据
3. 如果数据集中有其他列，可以在 prompt 中说明如何使用它们

### 示例 1：简单分类任务

**数据集格式**：
```csv
Headline,Typography
"标题1",1
"标题2",0
```

**Prompt 示例**：
```
你是一个文本分类助手。根据给定的标题，判断其 Typography 类别。

输入：标题文本
输出：只返回 0 或 1
```

### 示例 2：多列数据

**数据集格式**：
```csv
itemId,Headline,Category,Typography
id1,"标题1","新闻",1
id2,"标题2","娱乐",0
```

**Prompt 示例**：
```
你是一个文本分类助手。根据标题和类别信息，判断 Typography。

输入数据包含：
- Headline: 标题文本
- Category: 类别信息

请综合考虑标题和类别，输出 0 或 1。
```

### 示例 3：复杂数据结构

**数据集格式**：
```csv
id,question,context,answer
1,"问题1","上下文1","答案1"
2,"问题2","上下文2","答案2"
```

**Prompt 示例**：
```
你是一个问答助手。根据问题和上下文，生成答案。

输入：
- 问题：{question}
- 上下文：{context}

请根据上下文回答问题。
```

## 数据格式支持

### CSV 格式

- 支持逗号（`,`）和制表符（`\t`）分隔
- 自动检测分隔符
- 第一行应为列名

### JSON 格式

```json
[
  {
    "input": "输入文本",
    "ground_truth": "正确答案"
  }
]
```

### JSONL 格式

```jsonl
{"input": "输入文本1", "ground_truth": "答案1"}
{"input": "输入文本2", "ground_truth": "答案2"}
```

## 最佳实践

1. **使用标准列名**（推荐）：
   - 输入列使用 `input` 或 `text`
   - 答案列使用 `ground_truth` 或 `label`

2. **在 Prompt 中明确说明**：
   - 说明输入数据的格式
   - 说明如何使用数据中的各个字段
   - 说明期望的输出格式

3. **处理多列数据**：
   - 如果数据集中有多个相关列，在 prompt 中说明如何使用它们
   - 系统会保留所有列的数据，可以在 prompt 中引用

4. **测试数据加载**：
   - 上传数据集后，系统会显示数据预览
   - 检查预览是否正确识别了输入和答案列

## 注意事项

1. **列名不区分大小写**：`Headline` 和 `headline` 会被识别为同一列

2. **缺失值处理**：如果某列的值为空，系统会使用空字符串

3. **额外列保留**：数据集中除了输入和答案列之外的其他列会被保留，可以在 prompt 中使用

4. **没有标准答案**：如果没有 `ground_truth` 列，系统仍会运行，但无法评估准确率，优化过程会基于其他指标

## 示例：你的 example_dataset.csv

你的数据集格式：
```csv
itemId	Headline	Typography
id1	"标题1"	1
```

系统会自动识别：
- `Headline` → 作为输入列（匹配 `headline`）
- `Typography` → 作为标准答案列（匹配 `typography`）
- `itemId` → 保留但不使用

Prompt 只需要说明：
```
根据标题（Headline）预测 Typography 类别（0 或 1）
```

系统会自动将 `Headline` 列的值作为输入传递给模型。

