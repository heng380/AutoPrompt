# 配置指南

## API Key 配置

本项目支持两种 LLM 服务：
1. **OpenAI**（标准 OpenAI API）
2. **Azure OpenAI**（Azure 上的 OpenAI 服务）

你只需要配置其中一种即可。

---

## 方式一：使用 OpenAI（标准 OpenAI API）

### 步骤 1：获取 OpenAI API Key

1. 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
2. 登录你的 OpenAI 账号（如果没有账号，需要先注册）
3. 点击 "Create new secret key" 按钮
4. 给 Key 起个名字（可选）
5. 复制生成的 API Key（格式类似：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`）
   - ⚠️ **重要**：API Key 只显示一次，请立即保存！

### 步骤 2：配置 API Key

#### 方法一：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件：

```bash
cd autoprompt-langgraph
touch .env
```

编辑 `.env` 文件，添加：

```
OPENAI_API_KEY=sk-你的实际API密钥
```

例如：
```
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```

#### 方法二：使用环境变量

```bash
export OPENAI_API_KEY=sk-你的实际API密钥
```

---

## 方式二：使用 Azure OpenAI（推荐用于企业用户）

### 步骤 1：获取 Azure OpenAI 配置信息

你需要从 Azure Portal 获取以下信息：

1. **API Key** (`AZURE_OPENAI_API_KEY`)
   - 在 Azure Portal 中，进入你的 Azure OpenAI 资源
   - 在 "Keys and Endpoint" 页面可以找到 API Key

2. **Endpoint** (`AZURE_OPENAI_ENDPOINT`)
   - 格式类似：`https://your-resource-name.openai.azure.com/`
   - 也在 "Keys and Endpoint" 页面

3. **部署名称** (`AZURE_OPENAI_DEPLOYMENT_NAME`) - 可选
   - 这是你在 Azure OpenAI 中创建的模型部署名称
   - 如果不设置，将使用默认的模型名称（如 `gpt-3.5-turbo`）

4. **API 版本** (`AZURE_OPENAI_API_VERSION`) - 可选
   - 默认值：`2024-02-15-preview`
   - 可以在 Azure Portal 的 API 参考中找到支持的版本

### 步骤 2：配置 Azure OpenAI

#### 方法一：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件：

```bash
cd autoprompt-langgraph
touch .env
```

编辑 `.env` 文件，添加以下内容：

```
# Azure OpenAI 配置
AZURE_OPENAI_API_KEY=你的Azure_API密钥
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=你的部署名称（可选，如 gpt-35-turbo）
AZURE_OPENAI_API_VERSION=2024-02-15-preview（可选）
```

**完整示例**：
```
AZURE_OPENAI_API_KEY=abc123def456ghi789jkl012mno345pqr678
AZURE_OPENAI_ENDPOINT=https://my-openai-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

#### 方法二：使用环境变量

```bash
export AZURE_OPENAI_API_KEY=你的Azure_API密钥
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT_NAME=你的部署名称
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 步骤 3：验证配置

启动应用后：
- 如果看到 `✓ 使用 Azure OpenAI`，说明 Azure 配置成功
- 如果看到 `✓ 使用 OpenAI`，说明使用标准 OpenAI
- 如果看到警告信息，说明配置有问题

**注意**：
- 如果同时配置了 OpenAI 和 Azure OpenAI，系统会**优先使用 Azure OpenAI**
- `.env` 文件已被 `.gitignore` 忽略，不会被提交到 Git

---

## 常见问题

### 配置相关

**Q: 如何检查 API Key 是否配置成功？**
A: 启动应用时，如果没有看到警告信息，说明配置成功。或者可以在 Python 中测试：
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))  # 检查 OpenAI
print(os.getenv('AZURE_OPENAI_API_KEY'))  # 检查 Azure
```

**Q: 同时配置了 OpenAI 和 Azure OpenAI，会使用哪个？**
A: 系统会**优先使用 Azure OpenAI**。如果配置了 Azure 的相关参数，就会使用 Azure。

**Q: Azure OpenAI 的部署名称在哪里找？**
A: 在 Azure Portal 中，进入你的 Azure OpenAI 资源 → "Deployments" 页面，可以看到所有部署的名称。

**Q: Azure OpenAI 的 Endpoint 格式是什么？**
A: 格式为 `https://<your-resource-name>.openai.azure.com/`，注意末尾的斜杠是可选的。

### 安全相关

**Q: API Key 泄露了怎么办？**
A: 
- **OpenAI**: 立即到 [OpenAI Platform](https://platform.openai.com/api-keys) 删除该 Key，然后创建新的 Key
- **Azure OpenAI**: 在 Azure Portal 中重新生成 Key

**Q: .env 文件会被提交到 Git 吗？**
A: 不会，`.env` 文件已在 `.gitignore` 中，不会被提交到 Git。

### 使用相关

**Q: 可以使用其他 LLM 服务吗？**
A: 可以，但需要修改 `config.py` 和 `agents/` 目录下的 Agent 实现。

**Q: API Key 有使用限制吗？**
A: 
- **OpenAI**: 有使用限制和费用，请查看 [OpenAI 定价页面](https://openai.com/pricing)
- **Azure OpenAI**: 根据你的 Azure 订阅和配额设置，请查看 Azure Portal 中的配额信息

