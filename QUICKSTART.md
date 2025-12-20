# 快速开始指南

## 1. 安装依赖

```bash
cd autoprompt-langgraph
pip install -r requirements.txt
```

## 2. 配置 API Key

有两种方式配置 OpenAI API Key：

### 方式一：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件：

```bash
cd autoprompt-langgraph
cp .env.example .env
```

然后编辑 `.env` 文件，将 `your_openai_api_key_here` 替换为你的实际 API Key：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 方式二：使用环境变量

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**注意**：如果使用环境变量，需要在每次启动前都设置一次，或者将其添加到 `~/.bashrc` 或 `~/.zshrc` 中。

## 3. 启动服务

```bash
python app.py
```

## 4. 访问网站

打开浏览器访问：http://localhost:5000

## 5. 使用步骤

1. **上传数据集**
   - 点击上传区域或拖拽文件
   - 可以使用项目中的 `example_dataset.csv` 作为测试

2. **输入 Prompt**
   例如：
   ```
   你是一个数学计算助手。请根据用户的问题，计算出正确的答案。
   只返回数字结果，不要包含其他文字。
   ```

3. **设置迭代次数**
   - 默认 5 次，可以根据需要调整

4. **开始优化**
   - 点击"开始优化"按钮
   - 等待优化完成（可能需要几分钟）

5. **查看结果**
   - 查看优化后的 prompt
   - 查看准确率提升情况
   - 查看每轮迭代的改进建议

## 测试数据集

项目包含一个示例数据集 `example_dataset.csv`，包含简单的数学计算问题，可以用来测试系统。

## 常见问题

**Q: 提示 API Key 错误？**
A: 确保在 `.env` 文件中设置了正确的 `OPENAI_API_KEY`

**Q: 上传文件失败？**
A: 确保文件格式为 CSV、JSON 或 JSONL，且包含 `input` 和 `ground_truth` 字段

**Q: 优化时间很长？**
A: 这是正常的，因为需要对每个样本进行预测，并调用 LLM 进行分析和改写。建议先用小数据集测试。

