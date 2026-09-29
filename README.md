# Trial2 数学推理智能体

Trial2 是面向数学竞赛题的多阶段推理 Agent。评测平台从仓库根目录加载
`user_agent.py`，注入官方 Client，并调用 `ReasoningAgent.solve(problem, metadata)`。

## 项目结构

```text
Trial2/
├─ user_agent.py                 # 平台必需的根入口
├─ config.py                     # 模型、检索、温度和时间预算
├─ requirements.txt              # Python 依赖
├─ .gitignore                    # 本地缓存与日志忽略规则
├─ data/
│  └─ database/                  # Chroma 向量库及 HNSW 索引
├─ resources/
│  ├─ models/                    # Qwen3-Embedding-0.6B 等本地模型
│  └─ skills/                    # 数学领域技能文档与验证示例
└─ src/
   ├─ agent/                     # Agent 入口和总流程
   ├─ graph/                     # LangGraph 解题子图
   ├─ nodes/                     # 输入、分类、检索、推理、验证、协调节点
   ├─ state/                     # 跨节点状态定义
   ├─ mcp_servers/               # Python/SymPy 验证服务
   └─ utils/                     # 检索、答案整理、预算和日志工具
```

## 运行流程

1. 根入口加载 `src/` 并暴露平台要求的 `ReasoningAgent` 类。
2. Agent 初始化技能加载器、Python 验证服务和 Chroma 客户端。
3. 每道题经过输入处理、学科分类、混合检索、推理或计算验证，最后由协调器整理答案。
4. `solve()` 返回非空的 `final_response`，并附带可序列化的结构化 `trace`。

## 数据与模型

- `data/database/` 保存题目、解答和元数据，运行时只读检索。
- `resources/models/` 提供本地嵌入模型，向量检索不依赖互联网。
- `resources/skills/` 提供 18 个数学领域的方法、条件和验证模板。
- 大型数据库和模型文件按 Git LFS 规则管理。

## 提交约束

- 评测平台只加载根目录 `user_agent.py`，不需要修改调用方式。
- 所有资源路径均相对于项目目录解析，不依赖本机绝对路径。
- Client、密钥、限流和 Token 统计由平台注入，仓库不保存密钥。
- `trace` 只记录流程摘要、检索数量、推理结构和校验状态，不写入敏感信息。
