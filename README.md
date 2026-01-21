# 中文医疗知识图谱问答系统 (KGQA)

## 项目概述

本项目实现了一个完整的中文医疗领域知识图谱构建与智能问答系统，涵盖从知识抽取、融合、存储到问答应用的全链路技术。系统能够从非结构化医疗文本中自动抽取实体和关系，构建结构化知识图谱，并支持自然语言问答查询。

### 核心特性

- **实体识别与抽取**：基于BERT的中文医学命名实体识别
- **关系抽取**：从文本中自动抽取实体间的语义关系
- **知识融合与对齐**：多源知识整合，实体对齐与冲突消解
- **图数据库存储**：基于Neo4j的高效知识图谱存储与查询
- **知识推理引擎**：规则推理与链接预测
- **自然语言问答**：将自然语言问题转换为图查询
- **版本管理**：知识图谱的动态更新与回滚
- **可视化界面**：交互式知识图谱展示

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Streamlit UI   │  │   REST API      │  │  Swagger Docs   │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼─────────┐
│                         应用服务层                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  KGQA Service   │  │  Eval Service   │  │  Version Mgmt   │  │
│  │  (问答服务)      │  │  (评测服务)      │  │  (版本管理)      │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼─────────┐
│                         核心处理层                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │   NER    │ │    RE    │ │ Fusion   │ │Reasoning │           │
│  │ 实体识别  │ │ 关系抽取  │ │ 知识融合  │ │ 推理引擎  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                         数据存储层                               │
│                    ┌─────────────────┐                          │
│                    │     Neo4j       │                          │
│                    │   知识图谱数据库  │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI | 高性能异步Web框架 |
| **图数据库** | Neo4j 5.x | 原生图数据库，支持Cypher查询 |
| **NLP模型** | Transformers + PyTorch | 基于BERT的预训练模型 |
| **前端界面** | Streamlit | Python原生Web应用框架 |
| **可视化** | Pyvis + vis.js | 交互式网络图可视化 |
| **数据处理** | Pydantic | 数据验证与序列化 |
| **模糊匹配** | RapidFuzz | 高性能字符串相似度计算 |

---

## 项目结构

```
kgqa/
├── app/                        # FastAPI 应用模块
│   ├── main.py                 # 应用入口
│   ├── api.py                  # API路由定义
│   ├── schemas.py              # Pydantic数据模型
│   ├── kgqa_service.py         # 问答核心服务
│   └── neo4j_client.py         # Neo4j客户端封装
│
├── nlp/                        # 自然语言处理模块
│   ├── ner/                    # 命名实体识别
│   │   ├── extractor.py        # NER抽取器
│   │   └── schemas.py          # 实体数据结构
│   └── relation/               # 关系抽取
│       ├── extractor.py        # RE抽取器
│       └── schemas.py          # 三元组数据结构
│
├── fusion/                     # 知识融合模块
│   ├── pipeline.py             # 融合管道主流程
│   ├── normalize.py            # 文本规范化
│   ├── linking.py              # 实体链接与对齐
│   ├── conflict.py             # 冲突检测
│   └── scoring.py              # 边评分
│
├── reasoning/                  # 推理引擎模块
│   ├── engine.py               # 推理引擎
│   └── rules.py                # 推理规则定义
│
├── ingest/                     # 数据导入模块
│   ├── loader.py               # 数据加载器
│   ├── neo4j_importer.py       # Neo4j导入器
│   └── versioning.py           # 版本管理
│
├── eval/                       # 评测模块
│   ├── generator.py            # 评测集生成
│   ├── runner.py               # 评测运行器
│   ├── metrics.py              # 评测指标
│   └── report.py               # 报告生成
│
├── ui/                         # 用户界面模块
│   ├── streamlit_app.py        # Streamlit主应用
│   └── viz.py                  # 图谱可视化
│
├── data/                       # 数据目录
│   ├── triples_raw/            # 原始三元组数据
│   ├── triples_fused/          # 融合后数据
│   └── eval/                   # 评测数据
│
├── reports/                    # 评测报告目录
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖列表
├── run_pipeline.py             # 数据处理管道入口
└── run_eval.py                 # 评测入口
```

---

## 功能模块详解

### 1. 实体识别与抽取 (NER)

**文件位置**: `nlp/ner/extractor.py`

从非结构化中文医疗文本中识别和抽取实体，支持以下实体类型：

| 实体类型 | 标签 | 示例 |
|----------|------|------|
| 疾病 | Disease | 糖尿病、高血压、冠心病 |
| 症状 | Symptom | 头痛、发热、咳嗽 |
| 药物 | Drug | 阿司匹林、二甲双胍 |
| 检查 | Exam | 血常规、CT、心电图 |

**核心实现**：
- 支持基于Transformers的BERT模型
- 提供规则回退机制（无GPU时使用）
- 自动合并相邻同类实体

```python
from nlp.ner import NERExtractor

ner = NERExtractor(model_name="bert-base-chinese")
entities = ner.extract("糖尿病患者常见症状包括多饮、多尿和体重减轻")
# 输出: [EntitySpan(text="糖尿病", label="Disease"),
#        EntitySpan(text="多饮", label="Symptom"), ...]
```

### 2. 关系抽取 (RE)

**文件位置**: `nlp/relation/extractor.py`

识别实体间的语义关系，构建知识三元组。支持的关系类型：

| 关系类型 | 说明 | 示例 |
|----------|------|------|
| HAS_SYMPTOM | 疾病-症状 | (糖尿病, HAS_SYMPTOM, 多饮) |
| RECOMMENDED_DRUG | 疾病-药物 | (高血压, RECOMMENDED_DRUG, 氨氯地平) |
| NEEDS_EXAM | 疾病-检查 | (肺炎, NEEDS_EXAM, 胸部CT) |
| TREATS | 药物-疾病 | (胰岛素, TREATS, 糖尿病) |
| CONTRAINDICATED_FOR | 禁忌 | (阿司匹林, CONTRAINDICATED_FOR, 胃溃疡) |

**核心实现**：
- 基于模式匹配的规则抽取
- 支持模型推理（可选）
- 自动去重与置信度评估

```python
from nlp.relation import RelationExtractor

re = RelationExtractor()
triples = re.extract("糖尿病的治疗药物包括二甲双胍和胰岛素")
# 输出: [TripleRecord(head="糖尿病", relation="RECOMMENDED_DRUG", tail="二甲双胍"), ...]
```

### 3. 知识融合与对齐

**文件位置**: `fusion/pipeline.py`

整合多源知识，解决实体对齐和知识冲突问题。

**核心功能**：

1. **文本规范化** (`normalize.py`)
   - Unicode标准化
   - 全角转半角
   - 标点符号统一

2. **实体链接** (`linking.py`)
   - 基于规则的精确匹配
   - 基于RapidFuzz的模糊匹配
   - 别名抽取与管理

3. **冲突检测** (`conflict.py`)
   - 检测同一实体对的矛盾关系
   - 区分单值与多值关系

4. **边评分** (`scoring.py`)
   - 综合频率、来源权重、置信度计算边分数
   - 公式：`score = w1*freq + w2*source + w3*conf`

```python
from fusion import FusionPipeline

pipeline = FusionPipeline(config)
pipeline.load_triples("data/triples_raw/sample_triples.jsonl")
pipeline.run()
pipeline.export_fused_triples("data/triples_fused/fused_triples.jsonl")
```

### 4. 知识图谱存储

**文件位置**: `ingest/neo4j_importer.py`

基于Neo4j的高效图数据库存储方案。

**Schema设计**：

```
节点标签: Disease, Symptom, Drug, Exam, Indicator, Entity
关系类型: HAS_SYMPTOM, RECOMMENDED_DRUG, TREATS, NEEDS_EXAM,
         HAS_INDICATOR, CONTRAINDICATED_FOR, RELATED_TO

节点属性: name, aliases, created_at, updated_at
边属性: source, confidence, frequency, edge_score, conflict, predicted
```

**特性**：
- 唯一性约束保证数据一致性
- 全文索引支持模糊搜索
- MERGE操作实现幂等导入

### 5. 知识推理引擎

**文件位置**: `reasoning/engine.py`

基于规则的知识推理与链接预测。

**内置推理规则**：

| 规则ID | 描述 | 推理模式 |
|--------|------|----------|
| r1 | 症状缓解推理 | Disease→Drug ∧ Disease→Symptom ⇒ Drug→Symptom |
| r2 | 疾病关联推理 | Disease1→Symptom ∧ Disease2→Symptom (≥3) ⇒ Disease1↔Disease2 |
| r3 | 指标链推理 | Disease→Exam→Indicator ⇒ Disease→Indicator |
| r4 | 禁忌传播推理 | Drug⊗Disease1 ∧ Disease1↔Disease2 ⇒ Drug?⊗Disease2 |

```python
from reasoning import ReasoningEngine

engine = ReasoningEngine(neo4j_client)
inferences = engine.run_rules()
engine.write_predictions(inferences)
```

### 6. 自然语言问答

**文件位置**: `app/kgqa_service.py`

将自然语言问题转换为Cypher查询并返回结构化答案。

**处理流程**：

```
用户问题 → 意图识别 → 实体抽取 → Cypher生成 → 查询执行 → 答案构建
```

**支持的问句类型**：

| 意图 | 示例问题 |
|------|----------|
| disease_to_symptom | "糖尿病有哪些症状？" |
| symptom_to_disease | "头痛可能是什么病？" |
| disease_to_drug | "高血压吃什么药？" |
| disease_to_exam | "肺炎需要做什么检查？" |
| entity_definition | "什么是冠心病？" |

```python
from app.kgqa_service import KGQAService

service = KGQAService(neo4j_client)
response = service.ask("糖尿病有哪些症状？")
# 返回: AskResponse(intent="disease_to_symptom",
#                   answers=[AnswerItem(text="多饮"), ...],
#                   subgraph=SubGraph(...))
```

### 7. 版本管理

**文件位置**: `ingest/versioning.py`

实现知识图谱的动态更新与版本控制。

**核心功能**：
- **变更集管理**: 每次修改创建ChangeSet节点
- **审计日志**: 记录所有变更的AuditEvent
- **回滚支持**: 基于change_id回滚变更
- **历史查询**: 查看实体的变更历史

```python
from ingest.versioning import GraphVersioning

versioning = GraphVersioning(neo4j_client)
change_id = versioning.begin_changeset(actor="admin", reason="导入新数据")
versioning.upsert_triple(triple, change_id)
# 回滚
versioning.rollback(change_id)
```

### 8. 可视化展示

**文件位置**: `ui/viz.py`, `ui/streamlit_app.py`

基于Pyvis的交互式知识图谱可视化。

**功能特性**：
- 节点按类型着色（疾病红色、症状蓝色、药物绿色）
- 边标签显示关系类型和分数
- 支持缩放、拖拽、悬停提示
- 物理引擎自动布局

---

## 安装与部署

### 环境要求

- Python 3.10+
- Neo4j 5.x
- Docker (可选，用于运行Neo4j)

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd kgqa

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动Neo4j
docker run --name neo4j-kgqa -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j_password \
  neo4j:5

# 5. 配置数据库连接
# 编辑 config.yaml 中的 neo4j 配置
```

### 运行项目

```bash
# 步骤1: 运行数据处理管道
python run_pipeline.py

# 步骤2: 启动API服务
python app/main.py

# 步骤3: 启动UI界面（新终端）
streamlit run ui/streamlit_app.py
```

### 访问地址

| 服务 | 地址 |
|------|------|
| API文档 | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Neo4j Browser | http://localhost:7474 |

---

## API 接口文档

### 问答接口

**POST** `/api/ask`

请求体：
```json
{
  "query": "糖尿病有哪些症状？",
  "top_k": 10
}
```

响应：
```json
{
  "intent": "disease_to_symptom",
  "parsed_entities": [
    {"text": "糖尿病", "canonical": "糖尿病", "type": "Disease"}
  ],
  "cypher": "MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) ...",
  "answers": [
    {"text": "多饮", "score": 0.95, "source": ["textbook"], "conflict": false},
    {"text": "多尿", "score": 0.92, "source": ["textbook"], "conflict": false}
  ],
  "subgraph": {
    "nodes": [{"id": "糖尿病", "label": "糖尿病", "type": "Disease"}],
    "edges": [{"source": "糖尿病", "target": "多饮", "relation": "HAS_SYMPTOM"}]
  },
  "warnings": ["本系统仅用于知识检索与学习参考，不构成医疗建议。"],
  "latency_ms": 45
}
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats` | 获取图谱统计 |
| POST | `/api/eval/generate` | 生成评测集 |
| POST | `/api/eval/run` | 运行评测 |
| GET | `/api/eval/report` | 获取评测报告 |
| GET | `/api/health` | 健康检查 |

---

## 数据格式

### 输入数据格式

三元组数据采用JSONL格式，每行一个JSON对象：

```json
{"head": "糖尿病", "relation": "HAS_SYMPTOM", "tail": "多饮", "source": "textbook", "confidence": 0.95}
{"head": "糖尿病", "relation": "RECOMMENDED_DRUG", "tail": "二甲双胍", "source": "guideline", "confidence": 0.90}
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| head | string | 是 | 头实体名称 |
| relation | string | 是 | 关系类型 |
| tail | string | 是 | 尾实体名称 |
| source | string | 否 | 数据来源 |
| confidence | float | 否 | 置信度(0-1) |

### 支持的关系类型

| 关系 | 头实体类型 | 尾实体类型 |
|------|-----------|-----------|
| HAS_SYMPTOM | Disease | Symptom |
| RECOMMENDED_DRUG | Disease | Drug |
| TREATS | Drug | Disease |
| NEEDS_EXAM | Disease | Exam |
| HAS_INDICATOR | Exam | Indicator |
| CONTRAINDICATED_FOR | Drug | Disease |

---

## 评测指标

系统提供自动化评测功能，评估问答系统的性能：

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision | TP / (TP + FP) | 预测答案中正确的比例 |
| Recall | TP / (TP + FN) | 正确答案被召回的比例 |
| F1 | 2 * P * R / (P + R) | 精确率和召回率的调和平均 |
| Intent Accuracy | 正确意图数 / 总数 | 意图识别准确率 |

运行评测：
```bash
python run_eval.py
```

---

## 配置说明

`config.yaml` 配置文件详解：

```yaml
neo4j:
  uri: "bolt://localhost:7687"    # Neo4j连接地址
  user: "neo4j"                    # 用户名
  password: "neo4j_password"       # 密码

fusion:
  fuzzy_threshold: 90              # 模糊匹配阈值(0-100)
  allow_multi_relations:           # 允许多值的关系
    - "HAS_SYMPTOM"
    - "RECOMMENDED_DRUG"
  source_weights:                  # 来源权重
    default: 1.0
    textbook: 1.2
    guideline: 1.5
  scoring:                         # 评分权重
    frequency_weight: 0.3
    source_weight: 0.3
    confidence_weight: 0.4

nlp:
  ner:
    model: "bert-base-chinese"     # NER模型
    device: "cpu"                  # 设备(cpu/cuda)
    min_score: 0.5                 # 最低置信度
  relation:
    model: null                    # RE模型(null使用规则)
    device: "cpu"
    min_score: 0.5

reasoning:
  enabled: true                    # 是否启用推理
  max_inferences: 1000             # 最大推理数

eval:
  samples_per_type: 50             # 每类评测样本数
  random_seed: 42
  output_dir: "reports"

api:
  host: "0.0.0.0"
  port: 8000
  default_top_k: 10
```

---

## 常见问题

### 1. Neo4j连接失败

**问题**: `ServiceUnavailable: Unable to connect`

**解决**:
- 确认Neo4j服务已启动
- 检查config.yaml中的连接配置
- 确认端口7687未被占用

### 2. 模型加载缓慢

**问题**: 首次运行NER时下载模型很慢

**解决**:
- 提前下载模型到本地缓存
- 或使用规则模式（设置model为null）

### 3. 内存不足

**问题**: PyTorch模型加载时OOM

**解决**:
- 使用CPU模式：设置`device: "cpu"`
- 减小批处理大小
- 使用更小的模型

---

## 扩展开发

### 添加新的关系类型

1. 在`ingest/neo4j_importer.py`的`RELATION_MAPPING`中添加映射
2. 在`nlp/relation/extractor.py`的`RELATION_PATTERNS`中添加抽取规则
3. 在`app/kgqa_service.py`的`CYPHER_TEMPLATES`中添加查询模板

### 添加新的推理规则

在`reasoning/rules.py`中添加新的Rule对象：

```python
Rule(
    rule_id="r_new",
    description="规则描述",
    cypher="MATCH ... RETURN ...",
    target_relation="NEW_RELATION",
)
```

### 自定义NER模型

实现新的抽取器类，继承相同接口：

```python
class CustomNERExtractor:
    def extract(self, text: str) -> list[EntitySpan]:
        # 自定义实现
        pass
```

---

## 参考资料

- [Neo4j Cypher手册](https://neo4j.com/docs/cypher-manual/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [Streamlit文档](https://docs.streamlit.io/)
- [Pyvis文档](https://pyvis.readthedocs.io/)

---

