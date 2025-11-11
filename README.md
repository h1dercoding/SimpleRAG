# SimpleRAG

## 项目简介

SimpleRAG 是一个极简的 RAG（Retrieval-Augmented Generation）样例工程，演示如何将文档库向量化并结合检索来增强下游问答或生成任务。代码以 Python 编写，包含文件加载、向量存储、检索与问答链的基本实现，适合作为学习和快速原型的起点。

## 主要功能

- 将本地文本文件加载并构建向量索引
- 基于向量检索检索相关上下文并用于问答
- 提供命令行与简单前端（如有）的运行示例
- 文件溯源

## 环境要求

- Python 3.8+
- Ollama环境
    - qwen2.5:7b
    - bge-m3:latest
- langchain

2. 准备文档：

将你要添加到向量库的文本放入 `knowledge_base/` 目录（仓库中已包含示例文件）。

3. 运行示例脚本：

- 后端：

```bash
python app.py
```

- 前端：

```bash
python front.py
```

## 联系与反馈

如需帮助或有建议，请在仓库 Issue 中描述复现步骤和期望行为。
