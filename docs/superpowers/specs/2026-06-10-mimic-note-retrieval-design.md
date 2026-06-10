# MIMIC Note 语义检索系统设计方案

**日期:** 2026-06-10  
**状态:** 已批准，待实现  
**作者:** AI Assistant + User  

---

## 1. 项目概述

### 1.1 目标
构建一个基于 MIMIC 临床 note 数据的语义检索系统，支持自然语言查询和条件筛选，通过 all-MiniLM-L6-v2 进行语义分块与向量化，并使用 kimi 大模型对检索结果进行重排序与信息抽取。

### 1.2 数据规模
- **当前阶段**: 1,000 条 MIMIC note 样本（Discharge Summary 为主）
- **未来扩展**: 支持数十万条完整 MIMIC 数据集

### 1.3 成功标准
1. 自然语言查询能召回语义相关的 note 片段
2. 条件筛选能精确过滤 metadata（如 note_type、subject_id）
3. LLM 重排序能有效提升 Top-K 结果的相关性
4. 信息抽取支持二分类判断和指标提取，且允许 null（不存在时返回 null，不编造）
5. 架构预留 Web 界面扩展接口

---

## 2. 架构设计

### 2.1 总体架构

```
输入层                处理层                    输出层
─────────────────────────────────────────────────────────────
MIMIC Excel          DataLoader (pandas)       ChromaDB 索引
(1000条 note)        SemanticChunker           (持久化存储)
                     (spaCy + MiniLM)
                     
用户查询             Embedder (all-MiniLM)     Embedding 向量
(自然语言/           VectorStore (ChromaDB)    
 条件筛选)           
                     Retriever (相似度搜索)     检索结果 (Top-K)
                     
LLM API Key          LLMProcessor (kimi)       LLM 重排序
                     - rerank                  信息抽取
                     - classify                (二分类/指标)
                     - extract
```

### 2.2 核心数据流

**首次运行（索引构建）:**
1. 加载 Excel → 标准化 DataFrame
2. 语义分块（spaCy 分句 → MiniLM 句向量 → 多信号融合切分）
3. 生成 chunk embeddings（all-MiniLM-L6-v2）
4. 存入 ChromaDB（持久化到 `./chroma_db/`）

**查询流程:**
1. 加载已有 ChromaDB 索引
2. 用户输入查询（自然语言 + 可选条件筛选）
3. 向量检索 Top-K（扩大召回，如 Top-50）
4. LLM 重排序（精排为 Top-10）
5. LLM 信息抽取（二分类 / 指标提取）
6. 返回结构化结果

---

## 3. 模块划分与接口

### 3.1 模块清单

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据加载器 | `data_loader.py` | 读取 Excel，输出标准化 DataFrame |
| 语义分块器 | `semantic_chunker.py` | spaCy 分句 → 语义分块 → 输出 chunk 列表 |
| 向量化器 | `embedder.py` | 调用 all-MiniLM-L6-v2 生成 embedding |
| 向量存储/检索 | `vector_store.py` | ChromaDB 封装：添加、查询、条件过滤 |
| LLM 处理器 | `llm_processor.py` | 调用 kimi API：重排序、信息抽取 |
| 主流程 | `pipeline.py` | 编排以上模块，提供统一查询入口 |

### 3.2 核心数据结构

#### Chunk（分块后的最小单元）

```python
{
    "chunk_id": str,          # 全局唯一，如 "10000032-DS-21-0"
    "note_id": str,           # 原始 note_id
    "subject_id": int,
    "hadm_id": int,
    "note_type": str,
    "chunk_index": int,       # 该 note 内的第几个 chunk
    "text": str,              # chunk 文本内容
    "token_count": int,       # 用于调试和监控
}
```

#### 检索结果

```python
{
    "chunk": Chunk,
    "distance": float,        # 向量距离（越小越相似）
}
```

#### LLM 处理请求（一次一个需求）

```python
{
    "query": str,             # 用户原始查询
    "chunks": List[Chunk],    # 检索到的 Top-K chunks
    "task_type": str,         # "rerank" | "binary_classification" | "extraction"
    "task_prompt": str,       # 该次任务的具体指令
}
```

### 3.3 模块接口（伪代码）

#### data_loader.py

```python
def load_notes(excel_path: str) -> pd.DataFrame:
    """读取 Excel，返回标准化 DataFrame，包含列: 
    note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
    """
```

#### semantic_chunker.py

```python
class SemanticChunker:
    def __init__(self, embedder_model: str = "all-MiniLM-L6-v2"):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer(embedder_model)
    
    def chunk(self, note_text: str, metadata: dict) -> List[Chunk]:
        """
        1. spaCy 分句
        2. 句级 embedding
        3. 相邻相似度计算（百分比法，底部 20%）
        4. 检测结构边界（空行、标题行）
        5. 多信号融合确定切分点
        6. 长度后处理（MIN=80, TARGET=250, MAX=512）
        7. 超长块在局部最低相似度处二次切分
        8. 返回 Chunk 列表
        """
```

#### embedder.py

```python
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """批量生成 embedding，shape: (batch_size, 384)"""
```

#### vector_store.py

```python
class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """将 chunks 和 embeddings 存入 collection"""
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[RetrievalResult]:
        """向量检索 + metadata 条件筛选"""
```

#### llm_processor.py

```python
class LLMProcessor:
    def __init__(self, api_key: str, model: str = "kimi-latest", base_url: str = "..."):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float, str]]:
        """LLM 重排序，返回 (chunk, relevance_score, reasoning)"""
    
    def classify(self, chunks: List[Chunk], criteria: str) -> List[Tuple[Chunk, bool, str]]:
        """二分类：判断 chunk 是否满足某条件，返回 (chunk, is_match, reasoning)"""
    
    def extract(self, chunks: List[Chunk], fields: List[str]) -> List[dict]:
        """
        信息抽取：从 chunk 中提取指定字段
        若字段不存在，返回 null（不是空字符串或 "not found"）
        返回: [{"chunk_id": "...", "extractions": {"field1": "value1", "field2": null}, "confidence": "high"}, ...]
        """
```

#### pipeline.py

```python
class NoteSearchPipeline:
    def __init__(self, config: PipelineConfig):
        self.data_loader = DataLoader()
        self.chunker = SemanticChunker(config.chunk_config)
        self.embedder = Embedder(config.model_name)
        self.store = VectorStore(config.persist_dir)
        self.llm = LLMProcessor(config.api_key, config.model)
    
    def build_index(self, excel_path: str):
        """首次运行：加载 → 分块 → 向量化 → 存储"""
    
    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: int = 10,
        enable_rerank: bool = True,
        enable_llm_extract: bool = False,
        extract_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        """统一的查询入口。当前脚本直接调用；未来 Web 界面也调用此方法。"""
```

---

## 4. 语义分块策略详解

### 4.1 信号融合框架

最终切分决策由 **三类信号** 加权融合决定：

| 信号类型 | 来源 | 权重 | 说明 |
|---------|------|------|------|
| 语义边界 | 相邻句 MiniLM cosine similarity（底部 20%） | 高 | 核心切分依据 |
| 结构边界 | 空行（`\n\n+`）、疑似标题行 | 中 | 辅助信号，与语义边界协同 |
| 长度约束 | MIN=80 / TARGET=250 / MAX=512 token | 强制 | 硬性约束，不可突破 |

### 4.2 完整处理流程

```
Step 1: spaCy 分句 → 句列表
           ↓
Step 2: MiniLM 句向量 → 相邻相似度矩阵
           ↓
Step 3: 取底部 20% 相似度位置 → 语义候选边界
           ↓
Step 4: 检测结构边界（空行、标题行）→ 结构候选边界
           ↓
Step 5: 边界融合（结构边界附近若有语义边界，优先采用语义边界）
           ↓
Step 6: 生成 raw chunks → 长度检查
           ↓
Step 7: 长度后处理
           ├─ <80 token: 向后合并（优先）或向前合并
           ├─ 80-512 token: 保留（理想状态接近 TARGET=250）
           └─ >512 token: 在 raw chunk 内部找局部最低相似度处二次切分
           ↓
Step 8: 最终 Chunk 列表
```

### 4.3 关键参数

- **`MIN_CHUNK_TOKENS=80`**: 低于 80 token 的 chunk 不具备独立语义完整性，必须合并
- **`TARGET_CHUNK_TOKENS=250`**: 理想 chunk 长度，合并/切分时优先让 chunk 接近此值
- **`MAX_CHUNK_TOKENS=512`**: all-MiniLM-L6-v2 的最优输入长度附近，超过则二次切分
- **`SIMILARITY_PERCENTILE=20`**: 取相邻句子相似度分布的底部 20% 作为候选切分点

### 4.4 结构边界检测规则

```python
def is_structural_boundary(prev_line: str, curr_line: str) -> bool:
    # 空行信号
    if curr_line.strip() == "":
        return True
    
    # 标题信号：短行（<60字符）、全大写或首字母大写、以冒号结尾或独立成行
    stripped = curr_line.strip()
    if len(stripped) < 60 and (
        stripped.isupper() or 
        stripped.endswith(':') or
        stripped in KNOWN_SECTION_KEYWORDS  # 如 "CHIEF COMPLAINT", "MEDICATIONS"
    ):
        return True
    
    return False
```

**边界融合规则**: 若结构边界与语义边界距离 ≤2 句，以语义边界为准；否则两者都作为候选边界。

### 4.5 二次切分策略

当 raw chunk 超过 MAX=512 token 时：
1. **不采用简单中间切开**（会切断语义连贯性）
2. **在 chunk 内部的句子相似度序列中，找局部最小值**
3. 确保切分点在该 chunk 内部也是语义转折处

---

## 5. 向量检索与条件筛选

### 5.1 ChromaDB Collection 设计

```python
collection = client.create_collection(
    name="mimic_notes",
    metadata={"hnsw:space": "cosine"},  # 使用 cosine 距离，与 MiniLM 一致
)
```

**每 chunk 一条记录：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `ids` | string | chunk_id，如 `"10000032-DS-21-0"` |
| `embeddings` | float[384] | MiniLM 向量 |
| `documents` | string | chunk 文本 |
| `metadatas` | dict | `{note_id, subject_id, hadm_id, note_type, note_seq, chunk_index}` |

### 5.2 检索 API

```python
def search(
    self,
    query_text: str = None,
    query_embedding: np.ndarray = None,
    top_k: int = 10,
    filters: Optional[dict] = None,
    note_ids: Optional[List[str]] = None,
) -> List[RetrievalResult]:
```

### 5.3 两种检索模式

#### 模式 A：自然语言检索

```python
query = "肝硬化伴腹水的治疗方案"
query_emb = embedder.encode([query])
results = store.search(query_embedding=query_emb, top_k=20)
```

#### 模式 B：条件筛选检索

```python
filters = {
    "$and": [
        {"note_type": {"$eq": "DS"}},
        {"subject_id": {"$eq": 10000032}},
    ]
}
results = store.search(
    query_embedding=query_emb,
    top_k=20,
    filters=filters,
)
```

**支持的过滤操作符**: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`

### 5.4 混合检索策略

```python
# 典型场景：在 "Discharge Summary" 中检索 "肝硬化腹水"
results = store.search(
    query_embedding=query_emb,
    top_k=50,  # 先扩大召回
    filters={"note_type": {"$eq": "DS"}},
)
# 再经 LLM 重排序精排 Top-10
```

### 5.5 检索结果格式

```python
[
    {
        "chunk_id": "10000032-DS-21-3",
        "text": "Ascites - p/w worsening abd distension...",
        "distance": 0.12,  # cosine 距离，越小越相似
        "metadata": {
            "note_id": "10000032-DS-21",
            "subject_id": 10000032,
            "hadm_id": 22595853,
            "note_type": "DS",
            "chunk_index": 3,
        }
    },
    ...
]
```

**设计决策**: 检索阶段不限制 chunk 长度，召回尽可能多的候选（Top-50），让 LLM 在重排序阶段做精排。metadata 中保留完整的病例标识信息，方便溯源原始 note。

---

## 6. LLM 处理层

### 6.1 核心原则：一次只处理一个需求

不将重排序、分类、提取混在一个 prompt 中，而是**分步调用**。原因：

| 维度 | 单步（混合 prompt） | 分步（独立调用） |
|------|-------------------|-----------------|
| 准确率 | 低，LLM 容易混淆任务 | 高，每个任务专注 |
| 可调试性 | 差，出错难以定位 | 好，每步可单独验证 |
| Token 成本 | 低（一次调用） | 高（多次调用） |
| 灵活性 | 差，任务耦合 | 好，可单独开关某任务 |

**选择分步**，因为医学场景对准确性要求高，且 1000 条数据的查询量不大，token 成本可接受。

### 6.2 任务 1：重排序（Rerank）

**输入**: 用户查询 + Top-K chunks（如 Top-20）  
**输出**: `List[(Chunk, relevance_score, reasoning)]`，`relevance_score` 范围 0-100

**Prompt 模板**:

```
Query: {query}

Below are {len(chunks)} text fragments retrieved from medical notes.
Rate each fragment's relevance to the query on a scale of 0-100.
Provide a brief justification for each score.

Fragments:
{format_chunks(chunks)}

Return JSON:
[
  {"chunk_id": "...", "score": 85, "reasoning": "..."},
  ...
]
```

### 6.3 任务 2：二分类（Classify）

**输入**: chunks + 分类标准（如 "该片段是否提到肝硬化失代偿"）  
**输出**: `List[(Chunk, is_match, reasoning)]`，`is_match` 为 bool

**Prompt 模板**:

```
Criteria: {criteria}

Determine if each fragment meets the criteria. Answer true/false with reasoning.

Fragments:
{format_chunks(chunks)}

Return JSON:
[
  {"chunk_id": "...", "is_match": true, "reasoning": "..."},
  ...
]
```

### 6.4 任务 3：指标提取（Extract）

**输入**: chunks + 字段列表（如 `["MELD评分", "利尿剂用药"]`）  
**输出**: `List[dict]`，包含提取值和置信度

**核心约束：允许 null**

若字段在片段中不存在，**必须返回 null**，不允许编造值或返回空字符串。

**Prompt 模板**:

```
Extract the following fields from each fragment: {fields}

CRITICAL: If a field is not mentioned in the fragment, return null (not empty string or "not found").

Fragments:
{format_chunks(chunks)}

Return JSON:
[
  {
    "chunk_id": "...",
    "extractions": {"field1": "value1", "field2": null},
    "confidence": "high"  // high | medium | low
  },
  ...
]
```

### 6.5 批处理优化

- 每次发送 **5-10 个 chunks** 给一个 prompt
- 要求 LLM 返回 JSON 数组，包含所有 chunks 的结果
- 如果总 token 超过模型限制，自动拆分为多个 batch

---

## 7. Web 扩展预留

### 7.1 当前脚本层入口

```python
# pipeline.py
class NoteSearchPipeline:
    """当前本地脚本的核心类，未来可直接包装为 API"""
    
    def __init__(self, config: PipelineConfig):
        ...
    
    def build_index(self, excel_path: str):
        """首次运行：加载 → 分块 → 向量化 → 存储"""
        ...
    
    def search(self, query: str, filters: Optional[dict] = None, ...) -> SearchResult:
        """统一的查询入口。当前脚本直接调用；未来 Web 界面也调用此方法。"""
        ...
```

### 7.2 未来扩展路径

#### 方案 A：Streamlit/Gradio（最快）
- 复用 `NoteSearchPipeline` 类
- 前端界面：输入框（查询）+ 侧边栏（条件筛选）+ 结果展示
- 改动量：~100 行前端代码

#### 方案 B：FastAPI + 任意前端
- 将 `NoteSearchPipeline` 包装为 REST API
- 端点：`POST /search`, `POST /build-index`, `GET /health`
- 改动量：~50 行 API 代码 + 前端

### 7.3 预留设计决策

- `PipelineConfig` 使用 dataclass，支持从 YAML/JSON 加载，方便 Web 配置
- `VectorStore` 的 `persist_dir` 可配置，支持多租户隔离
- `LLMProcessor` 的 prompt 模板外置为 `.j2` 文件，支持热更新

---

## 8. 依赖清单

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | >=1.5 | Excel 读取 |
| spacy | >=3.6 | 英文分句 |
| en_core_web_sm | spacy 官方 | spaCy 英文模型 |
| sentence-transformers | >=2.2 | all-MiniLM-L6-v2 |
| chromadb | >=0.4 | 向量数据库 |
| numpy | >=1.24 | 数值计算 |
| openai | >=1.0 | kimi API 调用（OpenAI SDK 兼容） |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| spaCy 分句在医学缩写处误切（如 "Dr.", "e.g."） | 分块质量下降 | spaCy 模型对常见缩写已有处理，可辅以自定义规则 |
| MiniLM 对长医学文本的语义理解不足 | 检索召回率下降 | 语义分块将长文本切为 250 token 左右的短块，缓解此问题 |
| LLM 在信息抽取时编造不存在的值 | 医学场景严重后果 | 强制 prompt 要求不存在时返回 null；人工抽查验证 |
| ChromaDB 在数十万条数据时性能下降 | 查询延迟增加 | 未来可迁移至 Milvus/Pinecone；当前 1000 条无此问题 |

---

## 10. 测试策略

| 测试类型 | 覆盖内容 | 方法 |
|---------|---------|------|
| 单元测试 | 每个模块的独立功能 | pytest，mock 外部依赖 |
| 集成测试 | 端到端 pipeline | 使用 10 条样本数据验证完整流程 |
| 重排序准确率测试 | LLM rerank 是否提升相关性 | 人工标注 50 条查询的相关性，对比向量检索 vs LLM 重排序的 NDCG@10 |
| 抽取准确率测试 | LLM extract 的精确率和召回率 | 人工标注 50 条 chunk 的字段值，对比 LLM 输出 |
| 回归测试 | 索引构建后，查询结果是否稳定 | 固定种子，多次运行对比结果一致性 |

---

## 11. 附录

### 11.1 文件结构

```
project/
├── data/
│   └── 无标题.xlsx              # 输入数据
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-06-10-mimic-note-retrieval-design.md  # 本文档
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载器
│   ├── semantic_chunker.py     # 语义分块器
│   ├── embedder.py             # 向量化器
│   ├── vector_store.py         # 向量存储/检索
│   ├── llm_processor.py        # LLM 处理器
│   ├── pipeline.py             # 主流程
│   └── config.py               # 配置管理
├── prompts/
│   ├── rerank.j2               # 重排序 prompt 模板
│   ├── classify.j2             # 二分类 prompt 模板
│   └── extract.j2              # 信息抽取 prompt 模板
├── chroma_db/                  # ChromaDB 持久化目录（.gitignore）
├── tests/
│   └── ...                     # 测试文件
├── requirements.txt
└── main.py                     # 脚本入口
```

### 11.2 术语表

| 术语 | 说明 |
|------|------|
| Chunk | 语义分块后的文本片段 |
| Embedding | 文本的向量表示，用于语义相似度计算 |
| Rerank | 重排序，对初步检索结果进行精细排序 |
| Metadata | 与 chunk 关联的结构化信息（如 subject_id, note_type） |
| NDCG | Normalized Discounted Cumulative Gain，信息检索评价指标 |
| RAG | Retrieval-Augmented Generation，检索增强生成 |

---

**文档版本:** v1.0  
**最后更新:** 2026-06-10
