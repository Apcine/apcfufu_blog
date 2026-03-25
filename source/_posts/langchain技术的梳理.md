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
    LangChain 是一个帮助你构建 LLM 应用的 全套工具集 。涉及到prompt 构建、LLM 接入、记忆管理、工具调用、RAG、智能体开发等模块。
- RAG架构开发
    RAG（Retrieval-Augmented Generation），检索增强生成，目的是减少大模型的幻觉，提升回答质量，其流程如图。
![RAG](../images/rag.png)
    用户提问后，模型会从本地的向量数据库去寻找向量方向相近的内容，然后将内容与用户的问题结合提示词一起交给大模型去生成结果。
- Agent架构开发
    如果只能给出文本，或者说对话的话，终究只是纸上谈兵的llm，而agent就是给其工具，让其能通过工具去完成任务（要区别于强化学习的agent）
![Agent](../images/agent.png)


### 1.2 LangChain的安装 
推荐使用PyCharm编译器，建议下载专业版，专业版可以远程连接云服务器；如果是新了解python学习的，建议先了解一下Anaconda创建py隔离环境，避免py环境之间的污染
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#如果不是国内镜像源，请先设置为国内镜像源，如果设置过了可以直接跳过
pip install langchain
#默认安装最新版，目前1.x版本与0.x版本有比较大的变化，也可使用conda install langchain 
```

### 1.3 LangChain的框架
```
#获取api和url的变量

#提示词模板

#上下文记忆

#检索向量数据库（需要先分割文档并建立向量数据库）

#生成内容或任务逻辑

#调用工具完成任务

```

---

## 二、Model I/O
### 2.1 调用模型
模型可以分为非对话模型，对话模型，嵌入模型。调用模型之前要要在环境变量或者配置文件配好申请到的大模型api和url
```python
#LangChain的“hello world”，以qwen模型为例
import os
import dotenv 
from langchain_openai import OpenAI
dotenv.load_dotenv() #可以添加参数 override=True 来清理旧缓存的环境变量
os.environ["DASHSCOPE_API_KEY"]=os.getenv("DASHSCOPE_API_KEY")
os.environ["DASHSCOPE_BASE_URL"]=os.getenv("DASHSCOPE_BASE_URL")
llm = ChatOpenAI(    #在这里调用的是ChatModels，既可以生成AImessage，也可以生成List或PromptValue
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
response = llm.invoke("你好，你是谁") #invoke输出内容为message型
print(response.context) #content可以只返回模型的回答内容，不加content也可以输出，但是会有换行符等其他内容
```
`ChatOpenAI`的必须参数为`base_url,api_key,model`,其他参数也有：
`temperature`：控制文本的“随机性”，取值范围为0-1，越高越“抽象”，越低越“保守”
`max_tokens`：限制文本生成的最大token数，不过qwen的调用好像不是这个参数

不同的模型调用请参考各官方的操作手册
`invoke()`方法为阻塞式输出，会一次性生成ai回答结果，如果想向ds等模型那样一点一点的生成结果，可以在模型中设置`streaming = True`或者使用`stream()`调用，对于多个`HumanMessage`的请求，可以使用`batch()`进行批量调用

### 2.2 message的类型
`SystemMessage` 为AI的行为规则或背景信息，比如“你是智能助手科塔娜”
`HumanMessage` 表示来自用户的输入
`AIMessage` 一般存储AI回复的内容
```python
messages = [
SystemMessage(content="你是一个擅长人工智能相关学科的专家"),
HumanMessage(content="请解释一下什么是机器学习？")
]
response = chat_model.invoke(messages)
print(response.content)
```
这里可以看到SystemMessage有点类似于提示词的功能
---

### 2.3 PromptTemplate
```python
from langchain_core.prompts import PromptTemplate

prompt_template =PromptTemplate.from_template(
    template="你是一个{role},你的名字叫{name}",   #定义提示词模板的字符串，包含文本与变量占位符
    partial_variables={"role":"美食家"}   #字典格式，提前定义一些变量名
)

prompt = prompt_template.format(name="料理鼠王") #format()方法，给变量赋值，并且返回提示词格式，可以用于调用llm
print(prompt)
```

### 2.4 ChatPromptTemplate
相比于PromptTemplate，ChatPromptTemplate是创建聊天消息列表的提示词模板，更适合处理多角色，多轮次的对话场景
```PYTHON
from langchain_core.prompts import ChatPromptTemplate
chat_prompt_template =ChatPromptTemplate(
    messages=[
        ("system","你是一个{role},你的名字叫{name}"),
        ("human","请你评价一下{food}")
    ],
    input_variables=["role","name","food"],
)
#给模板赋值
prompt = chat_prompt_template.format_prompt(input={"role":"美食家","name":"料理鼠王","food":"麻婆豆腐"})  #使用 format_messages() 方法，返回消息列
表
response = chat_model.invoke(prompt)
print(response)
```

### 2.4 输出解析器 
语言模型返回的内容通常都是字符串的格式（文本格式），但在实际AI应用开发过程中，往往希望model可以返回更直观、更格式化的内容，以确保应用能够顺利进行后续的逻辑处理。此时，LangChain提供的 输出解析器 就派上用场了。
LangChain有许多不同类型的输出解析器：
`StrOutputParser` ：字符串解析器
`JsonOutputParser` ：JSON解析器，确保输出符合特定JSON对象格式
`XMLOutputParser` ：XML解析器，允许以流行的XML格式从LLM获取结果
`CommaSeparatedListOutputParser` ：CSV解析器，模型的输出以逗号分隔，以列表形式返回输出
`DatetimeOutputParser` ：日期时间解析器，可用于将 LLM 输出解析为日期时间格式
`OutputFixingParser` ：输出修复解析器，用于自动修复格式错误的解析器，比如将返回的不符合预期格式的输出，尝试修正为正确的结构化数据（如 JSON）
```python
##使用StrOutputParser()
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
response = llm.invoke("请简短介绍什么是3A游戏")

#使用StrOutputParser()
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
str = parser.invoke(response)
print(type(str))
print(str)

#使用JsonOutputParser()
actor_query = "皮克斯工作室"
prompt = f"""请列出{actor_query}的优秀电影作品，**严格按照JSON格式返回**，格式要求：
{{
    "studio": "皮克斯工作室",
    "excellent_movies": ["玩具总动员", "寻梦环游记", "心灵奇旅", ...]
}}"""
response = llm.invoke(prompt)
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
json_result = parser.parse(response.content)
print(json_result)
```

## 三、Memory（记忆系统）

### 3.1 Memory 的核心概念
- 为什么需要 Memory
正常调用大模型是不会记住上下文的，也就是上一次调用大模型的对话内容下一次调用大模型时，大模型是不会“记得”的。




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
Chain：链，用于将多个组件（提示模板、LLM模型、记忆、工具等）连接起来，形成可复用的 工作流 ，完成复杂的任务。Chain 的核心思想是通过组合不同的模块化单元，实现比单一组件更强大的功能。

- `LLMChain` - 基础链
这个链至少包括一个提示词模板（PromptTemplate），一个语言模型（LLM 或聊天模型）。
- `SimpleSequentialChain` - 顺序链
最简单的顺序链，多个链 串联执行 ，每个步骤都有 单一 的输入和输出，一个步骤的输出就是下一个步骤的输入，无需手动映射。
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

## 十、资源

### 10.1 参考资源
- 官方文档
- GitHub 示例
- 社区生态（LangGraph、LangServe 等）

---

## 附录


---

*本文持续更新中，如有错误或补充欢迎指正。*
