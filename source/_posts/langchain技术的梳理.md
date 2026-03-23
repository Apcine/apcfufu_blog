---
title: LangChain 技术梳理：从入门到实践
date: 2026-03-19 16:44:43
tags:
  - LangChain
  - LLM
  - AI
  - Python
categories:
  - 技术笔记
---

> 本文系统梳理 LangChain 核心组件的使用方法与工作原理，按照 Agent 的工作流程划分为 Prompt Engineering、Memory、Tools、RAG 四个核心模块。

<!-- toc -->

---

## 一、概述

### 1.1 什么是 LangChain
- LangChain 的定位与核心价值
- 框架设计哲学
- 适用场景与局限性

### 1.2 核心架构
- Components（组件层）
- Chains（链式调用）
- Agents（智能体）
- 数据流示意图

---

## 二、Prompt Engineering

### 2.1 Prompt Template 基础
- `PromptTemplate` - 基础模板
- `ChatPromptTemplate` - 对话模板
- `FewShotPromptTemplate` - 少样本示例模板
- 模板变量与格式化

### 2.2 Messages 类型与角色
- `SystemMessage` / `HumanMessage` / `AIMessage`
- `FunctionMessage` / `ToolMessage`
- Message History 的管理

### 2.3 输出解析器 (Output Parsers)
- `StrOutputParser` - 字符串解析
- `JsonOutputParser` - JSON 结构化输出
- `PydanticOutputParser` - 模型验证输出
- 自定义 Parser 实现

### 2.4 示例选择与动态构建
- Example Selectors 的作用
- `SemanticSimilarityExampleSelector`
- 动态 Few-Shot 学习

---

## 三、Memory（记忆系统）

### 3.1 Memory 的核心概念
- 为什么需要 Memory
- Short-term vs Long-term Memory
- Memory 在 Chain 中的集成方式

### 3.2 对话记忆类型
- `ConversationBufferMemory` - 完整缓冲
- `ConversationBufferWindowMemory` - 窗口缓冲
- `ConversationSummaryMemory` - 摘要记忆
- `ConversationSummaryBufferMemory` - 混合模式
- `ConversationEntityMemory` - 实体追踪

### 3.3 Vector Store  backed Memory
- 基于向量数据库的记忆
- `VectorStoreRetrieverMemory`
- 语义检索与相关性过滤

### 3.4 自定义 Memory 实现
- `BaseMemory` 接口
- `BaseChatMemory` 扩展
- 持久化策略（Redis、数据库等）

---

## 四、Tools（工具调用）

### 4.1 Tool 基础定义
- `@tool` 装饰器
- `StructuredTool` - 结构化参数
- Tool 的 name、description 设计原则

### 4.2 内置工具集
- `Search` 工具（DuckDuckGo、Google）
- `Wikipedia` 查询
- `Python REPL` 代码执行
- `Calculator` 计算工具
- API 调用工具

### 4.3 自定义工具开发
- 函数转 Tool 的最佳实践
- 异步 Tool 实现
- 错误处理与重试机制
- Tool 返回值的规范化

### 4.4 Tool Calling 机制
- Function Calling vs Tool Use
- OpenAI 函数调用规范
- 多工具并行执行
- 工具选择策略

---

## 五、RAG（检索增强生成）

### 5.1 RAG 架构概览
- 索引 (Indexing) → 检索 (Retrieval) → 生成 (Generation)
- Naive RAG vs Advanced RAG
- 流程图与数据流

### 5.2 Document 处理
- `Document` 对象结构
- 文档加载器 (Document Loaders)
  - 文本文件、PDF、网页
  - 数据库、API 数据源

### 5.3 文本分割策略
- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter`（推荐）
- `TokenTextSplitter`
- 基于语义的分割
- Chunk Size 与 Overlap 调优

### 5.4 向量化与存储
- Embedding Models（OpenAI、HuggingFace、本地模型）
- Vector Stores 对比
  - Chroma（轻量本地）
  - FAISS（Facebook）
  - Milvus / Pinecone（生产级）
- 索引持久化与加载

### 5.5 检索策略
- 相似度检索 (Similarity Search)
- MMR (Max Marginal Relevance) 多样性检索
- 元数据过滤
- 多查询检索 (Multi-Query)
- 重排序 (Reranking)

### 5.6 Retrieval Chain 构建
- `RetrievalQA` 基础链
- `ConversationalRetrievalChain`（带历史对话）
- 自定义 RAG Pipeline
- 检索结果注入 Prompt 的方式

---

## 六、Agent 系统

### 6.1 Agent 核心概念
- Agent = LLM + Tools + Memory + Planning
- ReAct 推理模式
- Plan-and-Execute 模式

### 6.2 Agent 类型对比
- `ZERO_SHOT_REACT_DESCRIPTION`
- `CHAT_ZERO_SHOT_REACT_DESCRIPTION`
- `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`
- OpenAI Functions Agent
- 如何选择合适的 Agent 类型

### 6.3 AgentExecutor
- 执行循环机制
- 最大迭代次数控制
- 错误处理与早停
- 回调与监控

---

## 七、Chains（链式组合）

### 7.1 基础 Chain 类型
- `LLMChain` - 基础链
- `SimpleSequentialChain` - 顺序链
- `SequentialChain` - 多输入输出链
- `RouterChain` - 路由链

### 7.2 LCEL (LangChain Expression Language)
- `|` 管道操作符
- `Runnable` 接口
- 链式组合最佳实践
- 并行执行与批处理

### 7.3 复杂工作流构建
- 条件分支
- 循环与递归
- 错误恢复机制

---

## 八、高级主题

### 8.1 回调与监控
- Callbacks 系统
- LangSmith 集成
- 日志与追踪

### 8.2 配置与部署
- 环境变量管理
- 模型切换策略
- 生产环境注意事项

### 8.3 性能优化
- 批处理 (Batching)
- 缓存策略
- 异步调用

---

## 九、实践案例

### 9.1 案例一：智能客服机器人
- 需求分析
- 架构设计
- 完整代码实现

### 9.2 案例二：知识库问答系统
- RAG Pipeline 搭建
- 多轮对话支持
- 效果评估与迭代

### 9.3 案例三：Multi-Agent 协作系统
- Agent 角色定义
- 任务分配与协作
- 结果汇总

---

## 十、总结与资源

### 10.1 学习路径建议
- 入门 → 进阶 → 精通
- 常见踩坑与解决方案

### 10.2 参考资源
- 官方文档
- GitHub 示例
- 社区生态（LangGraph、LangServe 等）

---

## 附录

### A. 版本兼容性说明
- LangChain 0.1.x vs 0.2.x 差异
- 迁移指南

### B. 快速参考表
- 常用类与方法速查
- 配置参数对照

---

*本文持续更新中，如有错误或补充欢迎指正。*
