# 中文医疗知识图谱 + 智能问答系统（KGQA）

> **核心目标**：实现一个“中文医疗领域知识图谱构建 + 智能问答”的完整项目，并在基础系统之上实现两个创新板块：**auto_fusion**（全自动融合对齐/冲突处理）与 **auto_eval**（全自动评测集生成 + 回归测试报告）。
>
> **核心要求**：创新板块必须可**完全用代码实现**，不依赖人工标注、人工审核或人为参与决策（除运行命令、填配置外）。

------

## 0. 运行环境与准备工作

- **开发/运行**：WSL2（Linux）中编写与运行 Python 代码
- **Windows 环境**：
  - Python 虚拟环境（Conda）已配置
  - CUDA 可用 (`torch.cuda.is_available()` 为 `True`)
  - Neo4j 已通过 Docker 启动

**Neo4j 配置参考：**

- **HTTP**: `http://localhost:7474`

- **Bolt**: `bolt://localhost:7687`

- **启动命令示例**：

  Bash

  ```
  docker run --name neo4j-kgqa -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neo4j_password \
    neo4j:5
  ```

------

## 1. 系统边界（总览）

本项目由 4 条主链路组成：

1. **数据 → 三元组（Triples）**
   - 输入：任意来源（数据集/文本/已有抽取结果）。
   - 输出：统一格式三元组 `(head_entity, relation, tail_entity, meta...)`。
   - *注：若实现信息抽取（NER/RE）为加分项，现阶段优先支持“直接导入 triples 文件”。*
2. **Triples → auto_fusion → Neo4j 图谱入库**
   - **核心步骤**：入库前必须经过 `auto_fusion` 处理（实体归一化、同义合并、冲突检测、质量字段补全）。
   - 结果：清洗后的数据存入 Neo4j，可通过 Cypher 查询。
3. **问答服务（KGQA）**
   - 提供 FastAPI 接口，支持：
     - 识别问题类型/意图。
     - 转换为 Cypher 模板查询 Neo4j。
     - 返回结构化结果（答案 + 路径/子图 + 证据/来源/冲突提示）。
4. **auto_eval 全自动评测与回归测试**
   - 直接从 Neo4j 自动生成评测集（问题 + 标准答案）。
   - 运行评测并输出报告（Markdown/HTML）。
   - 支持一键跑回归测试（用于图谱或代码更新后）。

------

## 2. 关键创新模块 A：auto_fusion（全自动融合对齐 + 冲突处理）

### 2.1 目标

在不依赖人工的前提下，将“脏、重复、别名、冲突”的三元组整理成可用的高质量 KG：

- 自动做 **实体归一化**（清洗规则）
- 自动做 **同义/别名合并**（规则 + 相似度）
- 自动做 **冲突关系检测与标记**
- 为节点/边补充通用质量字段（`source`, `confidence`, `created_at` 等）
- 问答查询时按可信度/一致性/频次进行排序输出（全自动）

### 2.2 输入与输出

- **输入**：原始 triples（JSONL/CSV，配置指定）
  - 最小字段：`head`, `relation`, `tail`
  - 推荐字段：`source`, `extractor`, `confidence` (0~1)
- **输出**：
  1. 融合后的规范 triples（用于入库）
  2. 实体对齐映射表（alias → canonical）
  3. 冲突检测报告（哪些 head+relation 存在多 tail 且互斥）

### 2.3 实体归一化（必须自动）

实现确定性规范化函数 `normalize_text()`，包含：

- 全半角统一、大小写统一
- 去除多余空格、统一标点
- 统一括号样式（`（）` 转 `()`）
- 去除常见噪声后缀/前缀（可配置，避免过拟合）

### 2.4 同义/别名合并（必须自动）

实现 `entity_linking()`：

1. **第一层（规则）**：normalize 后完全一致、括号别名、常见缩写等。
2. **第二层（相似度）**：
   - 建议：`rapidfuzz` 字符串相似度（无需 GPU）。
   - 可选：向量相似度（需作为可选开关）。
   - 阈值配置化（如 `FUZZY_THRESHOLD=90`）。
3. **输出**：Canonical 实体 + Alias 列表。

### 2.5 冲突检测（必须自动）

实现 `conflict_detection()`：

- 以 `(canonical_head, relation)` 为键，检测多个不同 tail。
- **天然多值关系**：配置 `ALLOW_MULTI=True`（如 `has_symptom`）。
- **互斥关系**：若同一 head 出现互斥 tail，标记 `conflict=true`。
- **策略**：保留所有候选，打分排序，查询时提示冲突。

### 2.6 可信度评分与排序（必须自动）

计算 `edge_score` 用于问答排序：

- **Frequency Score**：出现频次/多来源一致性。
- **Source Score**：来源权重。
- **Confidence Score**：抽取置信度。
- **输出**：按 `edge_score desc` 排序，保留 `source/confidence/frequency/conflict` 字段。

------

## 3. 关键创新模块 B：auto_eval（全自动评测集生成 + 回归测试报告）

### 3.1 目标

不依赖人工标注，构建可重复、可量化的评测体系：

- 从 Neo4j 自动抽样生成多类型问题。
- 标准答案由 Cypher 查询直接生成（Ground Truth）。
- 输出可读报告（Markdown/HTML），含指标与耗时。

### 3.2 评测题型（覆盖 1-hop/2-hop/反向）

需至少生成 4 类，每类 N 条（N 可配置）：

1. **1-hop 正向**：疾病 → 症状
2. **1-hop 反向**：症状 → 疾病
3. **2-hop 路径**：疾病 → 检查 → 指标
4. **属性/定义类**：实体简介/别名/分类

### 3.3 标准答案生成

对每条评测样本：

1. 生成 Cypher 查询。
2. 执行查询得到标准答案集合。
3. 存储样本：`question`, `intent`, `cypher`, `gold_answers`, `entity_mentions`。

### 3.4 评测指标

- **executable_rate**：是否生成可执行查询。
- **accuracy@1**：单答案准确率。
- **set_f1 / precision / recall**：集合答案（多值）指标。
- **coverage**：系统“可回答”比例。
- **latency**：p50/p95 响应时间（ms）。

### 3.5 输出报告

生成 `reports/latest_report.md`，包含：

- 总体指标表。
- 各题型分项指标。
- 最慢 Top-K 样本。
- 失败样本统计。

------

## 4. 基础系统：Neo4j Schema

Schema 需具备 Label 信息以支持 auto_eval。

### 4.1 节点（示例）

支持 Label 如 `:Disease`, `:Symptom`, `:Drug`, `:Exam`, `:Indicator`。

- 通用字段：`name` (canonical), `aliases` (list), `created_at`, `updated_at`。

### 4.2 关系（示例）

- `(:Disease)-[:HAS_SYMPTOM]->(:Symptom)`
- `(:Disease)-[:REQUIRES_EXAM]->(:Exam)`
- `(:Drug)-[:TREATS]->(:Disease)`
- `(:Drug)-[:CONTRAINDICATED_FOR]->(:Disease)`

**关系属性（由 auto_fusion 写入）：**

- `source`, `extractor`, `confidence`, `frequency`
- `conflict` (bool), `edge_score` (float)
- `predicted` (必须为 false)

------

## 5. KGQA 服务：FastAPI 接口契约

### 5.1 /ask（问答接口）

**输入：**

JSON

```
{
  "query": "糖尿病有哪些常见症状？",
  "top_k": 10,
  "mode": "facts_only"
}
```

**输出（必须结构化）：**

JSON

```
{
  "intent": "disease_to_symptom",
  "parsed_entities": [{"text":"糖尿病","canonical":"糖尿病","type":"Disease"}],
  "cypher": "MATCH ... RETURN ...",
  "answers": [{"text":"多饮", "score": 0.93, "source":["..."], "conflict": false}],
  "subgraph": { "nodes": [...], "edges": [...] },
  "warnings": ["本系统仅用于知识检索与学习参考，不构成医疗建议。", "检测到潜在冲突..."],
  "latency_ms": 123
}
```

### 5.2 auto_eval 接口

- `POST /eval/generate`：生成评测集（写入 `data/eval_set.jsonl`）。
- `POST /eval/run`：运行评测（调用 `/ask`）。
- `GET /eval/report`：返回最新报告内容。

------

## 6. Streamlit Demo

实现最小可视化界面：

1. **输入框**：输入自然语言问题。
2. **展示区**：
   - 答案列表（含 source/conflict/score）。
   - 子图可视化（网络图或列表）。
   - Warnings（免责声明 + 冲突提示）。
3. **评测区**：按钮运行 auto_eval 并展示报告摘要。

------

## 7. 工程要求

- **配置化**：使用 `.env` 或 `config.yaml` 管理：
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - `FUZZY_THRESHOLD`, `ALLOW_MULTI_RELATIONS`, `RANDOM_SEED`
- **可复现**：固定 Random Seed。
- **日志**：关键步骤（入库、合并、评测）打印清晰日志。
- **错误处理**：连接失败、空结果需返回明确信息。

------

## 8. 代码目录结构（建议）

Plaintext

```
kgqa/
├── app/
│   ├── main.py           # FastAPI 入口
│   ├── api.py            # 路由定义
│   ├── schemas.py        # Pydantic 模型
│   ├── kgqa_service.py   # 核心业务逻辑 (NL->Intent->Cypher->Answers)
│   └── neo4j_client.py   # Neo4j 连接工具
├── fusion/
│   ├── normalize.py      # 文本归一化
│   ├── linking.py        # 实体对齐/同义合并
│   ├── conflict.py       # 冲突检测
│   ├── scoring.py        # 评分逻辑
│   └── pipeline.py       # 融合主流程
├── ingest/
│   ├── loader.py         # 数据加载
│   ├── writer.py         # 中间文件输出
│   └── neo4j_importer.py # 入库脚本
├── eval/
│   ├── generator.py      # 评测集生成
│   ├── runner.py         # 评测执行器
│   ├── metrics.py        # 指标计算
│   └── report.py         # 报告生成
├── ui/
│   └── streamlit_app.py  # Demo 界面
├── data/
│   ├── triples_raw/      # 原始数据
│   ├── triples_fused/    # 融合后数据
│   └── eval/             # 评测数据与结果
├── reports/              # 生成的报告文件
├── config.yaml
├── .env.example
└── README.md
```

------

## 9. 安装依赖

在现有环境基础上，建议增加：

Bash

```
uv pip install rapidfuzz
# 可选：sentence-transformers (若启用向量相似度)
```

------

## 10. 验收标准

1. **图谱构建**：能从 Triples 输入构建 Neo4j 图谱，数据经清洗与融合。
2. **问答能力**：`/ask` 接口能回答至少 4 类问题，返回标准 JSON。
3. **Auto_fusion**：
   - [x] 包含 Normalize 功能。
   - [x] 包含自动同义合并（规则/相似度）。
   - [x] 包含冲突标记与评分排序。
4. **Auto_eval**：
   - [x] 自动生成评测集（含 Gold Answers）。
   - [x] 自动运行评测并产出报告（含指标统计）。
5. **Demo**：Streamlit 界面可运行，包含免责声明。