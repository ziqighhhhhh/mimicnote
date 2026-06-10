# MIMIC Note 语义检索系统实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现一个基于 MIMIC 临床 note 数据的语义检索系统，支持语义分块、向量检索、LLM 重排序和信息抽取。

**架构：** 模块化 Pipeline 设计（DataLoader → SemanticChunker → Embedder → VectorStore → LLMProcessor），使用 ChromaDB 持久化向量索引，all-MiniLM-L6-v2 生成语义向量，kimi LLM 进行重排序和信息抽取。

**技术栈：** Python 3.10+, pandas, spacy, sentence-transformers, chromadb, numpy, openai SDK (for kimi)

---

## 文件结构

```
project/
├── data/
│   └── 无标题.xlsx              # 输入数据（已存在）
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-06-10-mimic-note-retrieval-design.md
│       └── plans/
│           └── 2026-06-10-mimic-note-retrieval-plan.md    # 本文档
├── src/
│   ├── __init__.py
│   ├── config.py               # PipelineConfig dataclass
│   ├── data_loader.py          # 数据加载器
│   ├── semantic_chunker.py     # 语义分块器
│   ├── embedder.py             # 向量化器
│   ├── vector_store.py         # ChromaDB 封装
│   ├── llm_processor.py        # LLM 处理器
│   └── pipeline.py             # 主流程编排
├── prompts/
│   ├── rerank.j2               # 重排序 prompt 模板
│   ├── classify.j2             # 二分类 prompt 模板
│   └── extract.j2              # 信息抽取 prompt 模板
├── chroma_db/                  # ChromaDB 持久化目录（.gitignore）
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_semantic_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_llm_processor.py
│   └── test_pipeline.py
├── requirements.txt
├── .gitignore
└── main.py                     # CLI 入口
```

---

## 任务 1：项目初始化和依赖

**文件：**
- 创建：`requirements.txt`
- 创建：`.gitignore`
- 创建：`src/__init__.py`
- 创建：`tests/__init__.py`
- 创建：`prompts/rerank.j2`
- 创建：`prompts/classify.j2`
- 创建：`prompts/extract.j2`

- [ ] **步骤 1：创建 requirements.txt**

```txt
pandas>=1.5.0
spacy>=3.6.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
numpy>=1.24.0
openai>=1.0.0
pytest>=7.0.0
```

- [ ] **步骤 2：创建 .gitignore**

```gitignore
chroma_db/
__pycache__/
*.pyc
*.pkl
.env
.ipynb_checkpoints/
```

- [ ] **步骤 3：创建目录和空 __init__.py 文件**

```bash
mkdir -p src tests prompts
New-Item -ItemType File -Path "src/__init__.py" -Force
New-Item -ItemType File -Path "tests/__init__.py" -Force
```

- [ ] **步骤 4：安装依赖**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- [ ] **步骤 5：创建 prompt 模板文件**

`prompts/rerank.j2`:
```jinja2
Query: {{ query }}

Below are {{ chunks|length }} text fragments retrieved from medical notes.
Rate each fragment's relevance to the query on a scale of 0-100.
Provide a brief justification for each score.

Fragments:
{% for chunk in chunks %}
--- Fragment {{ loop.index }} ---
ID: {{ chunk.chunk_id }}
Text: {{ chunk.text }}
{% endfor %}

Return a JSON array:
[
  {"chunk_id": "...", "score": 85, "reasoning": "..."},
  ...
]
```

`prompts/classify.j2`:
```jinja2
Criteria: {{ criteria }}

Determine if each fragment meets the criteria. Answer true/false with reasoning.

Fragments:
{% for chunk in chunks %}
--- Fragment {{ loop.index }} ---
ID: {{ chunk.chunk_id }}
Text: {{ chunk.text }}
{% endfor %}

Return a JSON array:
[
  {"chunk_id": "...", "is_match": true, "reasoning": "..."},
  ...
]
```

`prompts/extract.j2`:
```jinja2
Extract the following fields from each fragment: {{ fields|join(", ") }}

CRITICAL: If a field is not mentioned in the fragment, return null (not empty string or "not found").

Fragments:
{% for chunk in chunks %}
--- Fragment {{ loop.index }} ---
ID: {{ chunk.chunk_id }}
Text: {{ chunk.text }}
{% endfor %}

Return a JSON array:
[
  {
    "chunk_id": "...",
    "extractions": {"field1": "value1", "field2": null},
    "confidence": "high"
  },
  ...
]
```

- [ ] **步骤 6：Commit**

```bash
git add requirements.txt .gitignore src/__init__.py tests/__init__.py prompts/
git commit -m "chore: project setup with dependencies and prompt templates"
```

---

## 任务 2：配置管理（config.py）

**文件：**
- 创建：`src/config.py`
- 测试：`tests/test_config.py`（内联验证，无需独立测试文件）

- [ ] **步骤 1：编写 config.py**

```python
"""配置管理模块。"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ChunkConfig:
    """语义分块配置。"""
    min_chunk_tokens: int = 80
    target_chunk_tokens: int = 250
    max_chunk_tokens: int = 512
    similarity_percentile: int = 20  # 底部 20% 作为候选切分点
    embedder_model: str = "all-MiniLM-L6-v2"


@dataclass
class PipelineConfig:
    """Pipeline 主配置。"""
    # 数据配置
    excel_path: str = "data/无标题.xlsx"
    
    # 分块配置
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    
    # 向量化配置
    model_name: str = "all-MiniLM-L6-v2"
    
    # 向量数据库配置
    persist_dir: str = "./chroma_db"
    collection_name: str = "mimic_notes"
    
    # LLM 配置
    api_key: Optional[str] = None
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "kimi-latest"
    
    # 检索配置
    default_top_k: int = 10
    rerank_top_n: int = 10
    retrieve_top_k: int = 50  # 扩大召回
    
    # Prompt 模板路径
    prompts_dir: str = "./prompts"
```

- [ ] **步骤 2：验证配置类可导入**

```bash
python -c "from src.config import PipelineConfig, ChunkConfig; c = PipelineConfig(); print(c.chunk_config.max_chunk_tokens)"
```

预期输出：`512`

- [ ] **步骤 3：Commit**

```bash
git add src/config.py
git commit -m "feat: add PipelineConfig and ChunkConfig dataclasses"
```

---

## 任务 3：数据加载器（data_loader.py）

**文件：**
- 创建：`src/data_loader.py`
- 测试：`tests/test_data_loader.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_data_loader.py`:
```python
"""DataLoader 测试。"""
import pytest
import pandas as pd
from src.data_loader import DataLoader


class TestDataLoader:
    def test_load_notes_returns_dataframe(self, tmp_path):
        # 创建测试用的 Excel 文件
        test_data = pd.DataFrame({
            "note_id": ["10000032-DS-21", "10000032-DS-22"],
            "subject_id": [10000032, 10000032],
            "hadm_id": [22595853, 22841357],
            "note_type": ["DS", "DS"],
            "note_seq": [21, 22],
            "charttime": pd.to_datetime(["2180-05-07", "2180-06-27"]),
            "storetime": pd.to_datetime(["2180-05-09 15:26:00", "2180-07-01 10:15:00"]),
            "text": ["Patient has ascites...", "Patient improved..."],
        })
        excel_path = tmp_path / "test.xlsx"
        test_data.to_excel(excel_path, index=False)
        
        loader = DataLoader()
        df = loader.load_notes(str(excel_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["note_id", "subject_id", "hadm_id", "note_type", 
                                     "note_seq", "charttime", "storetime", "text"]
    
    def test_load_notes_empty_text_filtered(self, tmp_path):
        test_data = pd.DataFrame({
            "note_id": ["A-1", "A-2"],
            "subject_id": [1, 1],
            "hadm_id": [1, 1],
            "note_type": ["DS", "DS"],
            "note_seq": [1, 2],
            "charttime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "storetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "text": ["Valid text", ""],  # 空文本应被过滤
        })
        excel_path = tmp_path / "test.xlsx"
        test_data.to_excel(excel_path, index=False)
        
        loader = DataLoader()
        df = loader.load_notes(str(excel_path))
        
        assert len(df) == 1
        assert df.iloc[0]["note_id"] == "A-1"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_data_loader.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.data_loader'`

- [ ] **步骤 3：编写最少实现代码**

`src/data_loader.py`:
```python
"""数据加载器模块。"""
import pandas as pd


class DataLoader:
    """加载 MIMIC note 数据。"""
    
    def load_notes(self, excel_path: str) -> pd.DataFrame:
        """读取 Excel，返回标准化 DataFrame。
        
        过滤掉 text 为空的记录。
        """
        df = pd.read_excel(excel_path)
        
        # 确保必要列存在
        required_columns = ["note_id", "subject_id", "hadm_id", "note_type", 
                           "note_seq", "charttime", "storetime", "text"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 过滤空文本
        df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_data_loader.py -v
```

预期：PASS，2 tests passed

- [ ] **步骤 5：Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: add DataLoader with empty text filtering"
```

---

## 任务 4：向量化器（embedder.py）

**文件：**
- 创建：`src/embedder.py`
- 测试：`tests/test_embedder.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_embedder.py`:
```python
"""Embedder 测试。"""
import pytest
import numpy as np
from src.embedder import Embedder


class TestEmbedder:
    def test_encode_returns_numpy_array(self):
        embedder = Embedder()
        texts = ["This is a test.", "Another test sentence."]
        embeddings = embedder.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 维度
    
    def test_encode_single_text(self):
        embedder = Embedder()
        embeddings = embedder.encode(["Single text"])
        
        assert embeddings.shape == (1, 384)
    
    def test_cosine_similarity_range(self):
        embedder = Embedder()
        embeddings = embedder.encode(["hello world", "hello world"])
        
        # 相同文本的相似度应接近 1
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(embeddings)[0, 1]
        assert sim > 0.99
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_embedder.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.embedder'`

- [ ] **步骤 3：编写最少实现代码**

`src/embedder.py`:
```python
"""向量化器模块。"""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """使用 all-MiniLM-L6-v2 生成文本 embedding。"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """批量生成 embedding。
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy array, shape: (len(texts), 384)
        """
        return self.model.encode(texts, convert_to_numpy=True)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_embedder.py -v
```

预期：PASS，3 tests passed（首次运行会下载模型，可能需要几分钟）

- [ ] **步骤 5：Commit**

```bash
git add src/embedder.py tests/test_embedder.py
git commit -m "feat: add Embedder with all-MiniLM-L6-v2"
```

---

## 任务 5：语义分块器（semantic_chunker.py）

**文件：**
- 创建：`src/semantic_chunker.py`
- 测试：`tests/test_semantic_chunker.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_semantic_chunker.py`:
```python
"""SemanticChunker 测试。"""
import pytest
from src.semantic_chunker import SemanticChunker


class TestSemanticChunker:
    @pytest.fixture
    def chunker(self):
        return SemanticChunker()
    
    def test_chunk_returns_list(self, chunker):
        text = "Patient has diabetes. He is on insulin. Blood sugar is high."
        metadata = {"note_id": "TEST-1", "subject_id": 1, "hadm_id": 1, 
                    "note_type": "DS", "note_seq": 1}
        chunks = chunker.chunk(text, metadata)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "token_count" in chunk
    
    def test_chunk_size_constraints(self, chunker):
        # 较长的文本应被分成多个 chunk
        text = "Patient has diabetes. " * 100  # 较长文本
        metadata = {"note_id": "TEST-2", "subject_id": 1, "hadm_id": 1,
                    "note_type": "DS", "note_seq": 1}
        chunks = chunker.chunk(text, metadata)
        
        for chunk in chunks:
            assert chunk["token_count"] <= 512  # MAX
    
    def test_chunk_preserves_metadata(self, chunker):
        text = "Simple text."
        metadata = {"note_id": "TEST-3", "subject_id": 999, "hadm_id": 888,
                    "note_type": "DS", "note_seq": 42}
        chunks = chunker.chunk(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0]["note_id"] == "TEST-3"
        assert chunks[0]["subject_id"] == 999
        assert chunks[0]["hadm_id"] == 888
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_semantic_chunker.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.semantic_chunker'`

- [ ] **步骤 3：编写实现代码**

`src/semantic_chunker.py`:
```python
"""语义分块器模块。

基于 spaCy 分句 + MiniLM 句向量 + 多信号融合的语义分块策略。
"""
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any


# 常见的 MIMIC note 章节标题关键词
KNOWN_SECTION_KEYWORDS = {
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HPI",
    "PAST MEDICAL HISTORY", "PMH", "MEDICATIONS", "ALLERGIES",
    "PHYSICAL EXAM", "PHYSICAL EXAMINATION", "VITALS",
    "LABORATORY DATA", "LABS", "IMAGING", "STUDIES",
    "ASSESSMENT", "PLAN", "DISPOSITION", "DISCHARGE DIAGNOSIS",
    "DISCHARGE MEDICATIONS", "FOLLOWUP", "FOLLOW UP",
    "BRIEF HOSPITAL COURSE", "HOSPITAL COURSE",
}


class SemanticChunker:
    """语义分块器。
    
    信号融合框架：
    1. 语义边界：相邻句 MiniLM cosine similarity（底部 20%）
    2. 结构边界：空行、疑似标题行
    3. 长度约束：MIN=80, TARGET=250, MAX=512 token
    """
    
    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        min_chunk_tokens: int = 80,
        target_chunk_tokens: int = 250,
        max_chunk_tokens: int = 512,
        similarity_percentile: int = 20,
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer(embedder_model)
        self.min_chunk_tokens = min_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.similarity_percentile = similarity_percentile
    
    def chunk(self, note_text: str, metadata: dict) -> List[Dict[str, Any]]:
        """将 note 文本切分为语义 chunk。
        
        Args:
            note_text: 原始 note 文本
            metadata: 包含 note_id, subject_id, hadm_id, note_type, note_seq
            
        Returns:
            Chunk 列表
        """
        # Step 1: spaCy 分句
        sentences = self._split_sentences(note_text)
        if not sentences:
            return []
        
        # Step 2: 句级 embedding
        sentence_embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        
        # Step 3: 相邻相似度
        similarities = self._compute_similarities(sentence_embeddings)
        
        # Step 4 & 5: 检测结构边界 + 融合
        structural_breaks = self._detect_structural_boundaries(note_text, sentences)
        semantic_breaks = self._detect_semantic_breaks(similarities)
        final_breaks = self._merge_boundaries(semantic_breaks, structural_breaks, len(sentences))
        
        # Step 6: 生成 raw chunks
        raw_chunks = self._build_raw_chunks(sentences, final_breaks)
        
        # Step 7: 长度后处理
        processed_chunks = self._apply_length_constraints(raw_chunks)
        
        # Step 8: 包装为 Chunk dict
        return self._wrap_chunks(processed_chunks, metadata)
    
    def _split_sentences(self, text: str) -> List[str]:
        """spaCy 分句，过滤空句。"""
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) >= 5:  # 过滤过短句子
                sentences.append(sent_text)
        return sentences
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """计算相邻句子的 cosine similarity。"""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities
    
    def _detect_structural_boundaries(self, text: str, sentences: List[str]) -> List[int]:
        """检测结构边界（空行、标题行），返回句子索引列表。"""
        breaks = []
        lines = text.split('\n')
        
        # 构建行到句子的映射
        line_to_sent_idx = {}
        char_offset = 0
        for sent_idx, sent in enumerate(sentences):
            # 找到句子在原始文本中的位置
            pos = text.find(sent, char_offset)
            if pos >= 0:
                # 计算这是第几行
                line_num = text[:pos].count('\n')
                line_to_sent_idx[line_num] = sent_idx
                char_offset = pos + len(sent)
        
        # 检测空行和标题行
        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # 空行
                if line_idx + 1 < len(lines):
                    # 找到空行后的第一个句子
                    for sent_idx in range(len(sentences)):
                        if sent_idx not in line_to_sent_idx.values():
                            continue
                    # 简化：空行后接的句子作为边界
                    for si, sj in line_to_sent_idx.items():
                        if si >= line_idx:
                            breaks.append(sj)
                            break
            elif self._is_title_line(stripped):
                # 标题行
                for si, sj in line_to_sent_idx.items():
                    if si >= line_idx:
                        breaks.append(sj)
                        break
        
        return sorted(set(b for b in breaks if b > 0 and b < len(sentences)))
    
    def _is_title_line(self, line: str) -> bool:
        """判断是否为标题行。"""
        if len(line) > 60:
            return False
        if line.isupper():
            return True
        if line.endswith(':'):
            return True
        if line in KNOWN_SECTION_KEYWORDS:
            return True
        return False
    
    def _detect_semantic_breaks(self, similarities: List[float]) -> List[int]:
        """基于百分比法检测语义边界。"""
        if not similarities:
            return []
        
        threshold = np.percentile(similarities, self.similarity_percentile)
        breaks = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        return breaks
    
    def _merge_boundaries(self, semantic_breaks: List[int], structural_breaks: List[int], 
                          num_sentences: int) -> List[int]:
        """融合语义边界和结构边界。"""
        # 结构边界附近若有语义边界（距离<=2），优先采用语义边界
        merged = set(semantic_breaks)
        
        for sb in structural_breaks:
            # 检查是否有语义边界在附近
            nearby = any(abs(sb - sem) <= 2 for sem in semantic_breaks)
            if not nearby:
                merged.add(sb)
        
        # 确保边界在有效范围内
        merged = {b for b in merged if 0 < b < num_sentences}
        
        return sorted(merged)
    
    def _build_raw_chunks(self, sentences: List[str], breaks: List[int]) -> List[List[str]]:
        """根据边界将句子分组。"""
        if not breaks:
            return [sentences]
        
        chunks = []
        start = 0
        for break_point in breaks:
            chunks.append(sentences[start:break_point])
            start = break_point
        chunks.append(sentences[start:])
        
        return [c for c in chunks if c]  # 过滤空组
    
    def _apply_length_constraints(self, raw_chunks: List[List[str]]) -> List[List[str]]:
        """应用长度约束：合并短块，切分长块。"""
        result = []
        i = 0
        
        while i < len(raw_chunks):
            chunk = raw_chunks[i]
            token_count = self._count_tokens(chunk)
            
            if token_count < self.min_chunk_tokens:
                # 尝试向后合并
                if i + 1 < len(raw_chunks):
                    merged = chunk + raw_chunks[i + 1]
                    merged_tokens = self._count_tokens(merged)
                    if merged_tokens <= self.max_chunk_tokens:
                        result.append(merged)
                        i += 2
                        continue
                # 无法合并则保留（可能是最后一个）
                if result and token_count < self.min_chunk_tokens:
                    # 尝试向前合并
                    prev = result[-1] + chunk
                    if self._count_tokens(prev) <= self.max_chunk_tokens:
                        result[-1] = prev
                        i += 1
                        continue
                result.append(chunk)
                i += 1
            
            elif token_count > self.max_chunk_tokens:
                # 在局部最低相似度处二次切分
                sub_chunks = self._split_long_chunk(chunk)
                result.extend(sub_chunks)
                i += 1
            
            else:
                result.append(chunk)
                i += 1
        
        return result
    
    def _split_long_chunk(self, chunk: List[str]) -> List[List[str]]:
        """对超长 chunk 在局部最低相似度处二次切分。"""
        if len(chunk) <= 2:
            return [chunk]
        
        embeddings = self.embedder.encode(chunk, convert_to_numpy=True)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        
        # 找局部最低相似度
        min_idx = np.argmin(similarities)
        split_point = min_idx + 1
        
        left = chunk[:split_point]
        right = chunk[split_point:]
        
        result = []
        for sub in [left, right]:
            if self._count_tokens(sub) > self.max_chunk_tokens and len(sub) > 2:
                result.extend(self._split_long_chunk(sub))
            else:
                result.append(sub)
        
        return result
    
    def _count_tokens(self, sentences: List[str]) -> int:
        """估算 token 数（使用 spaCy token 数）。"""
        text = " ".join(sentences)
        doc = self.nlp(text)
        return len(doc)
    
    def _wrap_chunks(self, sentence_groups: List[List[str]], metadata: dict) -> List[Dict[str, Any]]:
        """将句子组包装为 Chunk dict。"""
        chunks = []
        for idx, group in enumerate(sentence_groups):
            text = " ".join(group)
            token_count = self._count_tokens(group)
            chunk = {
                "chunk_id": f"{metadata['note_id']}-{idx}",
                "note_id": metadata["note_id"],
                "subject_id": metadata["subject_id"],
                "hadm_id": metadata["hadm_id"],
                "note_type": metadata["note_type"],
                "chunk_index": idx,
                "text": text,
                "token_count": token_count,
            }
            chunks.append(chunk)
        return chunks
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_semantic_chunker.py -v
```

预期：PASS，3 tests passed（首次运行可能需要下载 spacy 模型）

- [ ] **步骤 5：Commit**

```bash
git add src/semantic_chunker.py tests/test_semantic_chunker.py
git commit -m "feat: add SemanticChunker with multi-signal fusion"
```

---

## 任务 6：向量存储（vector_store.py）

**文件：**
- 创建：`src/vector_store.py`
- 测试：`tests/test_vector_store.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_vector_store.py`:
```python
"""VectorStore 测试。"""
import pytest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_test")
        return VectorStore(persist_dir=persist_dir)
    
    def test_add_and_query(self, store):
        chunks = [
            {
                "chunk_id": "TEST-1-0",
                "note_id": "TEST-1",
                "subject_id": 1,
                "hadm_id": 1,
                "note_type": "DS",
                "chunk_index": 0,
                "text": "Patient has diabetes.",
                "token_count": 10,
            },
            {
                "chunk_id": "TEST-1-1",
                "note_id": "TEST-1",
                "subject_id": 1,
                "hadm_id": 1,
                "note_type": "DS",
                "chunk_index": 1,
                "text": "Patient has hypertension.",
                "token_count": 10,
            },
        ]
        embeddings = np.array([[0.1] * 384, [0.2] * 384], dtype=np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        # 查询与第一个 chunk 相似的向量
        results = store.query(embeddings[0], top_k=2)
        assert len(results) == 2
        assert results[0]["chunk"]["chunk_id"] == "TEST-1-0"
    
    def test_query_with_filter(self, store):
        chunks = [
            {
                "chunk_id": "TEST-A-0",
                "note_id": "TEST-A",
                "subject_id": 1,
                "hadm_id": 1,
                "note_type": "DS",
                "chunk_index": 0,
                "text": "Diabetes note.",
                "token_count": 10,
            },
            {
                "chunk_id": "TEST-B-0",
                "note_id": "TEST-B",
                "subject_id": 2,
                "hadm_id": 2,
                "note_type": "Nursing",
                "chunk_index": 0,
                "text": "Nursing note.",
                "token_count": 10,
            },
        ]
        embeddings = np.array([[0.1] * 384, [0.9] * 384], dtype=np.float32)
        store.add_chunks(chunks, embeddings)
        
        # 过滤 note_type="DS"
        results = store.query(
            embeddings[1],
            top_k=10,
            filters={"note_type": {"$eq": "DS"}}
        )
        assert len(results) == 1
        assert results[0]["chunk"]["note_type"] == "DS"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_vector_store.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.vector_store'`

- [ ] **步骤 3：编写最少实现代码**

`src/vector_store.py`:
```python
"""向量存储/检索模块。

封装 ChromaDB，提供添加、查询、条件过滤功能。
"""
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb


class VectorStore:
    """ChromaDB 向量存储封装。"""
    
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "mimic_notes"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """将 chunks 和 embeddings 存入 collection。
        
        Args:
            chunks: Chunk 列表
            embeddings: numpy array, shape: (len(chunks), 384)
        """
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "note_id": c["note_id"],
                "subject_id": c["subject_id"],
                "hadm_id": c["hadm_id"],
                "note_type": c["note_type"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]
        
        # 转换为 list[list[float]]
        embeddings_list = embeddings.tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索 + 条件筛选。
        
        Args:
            query_embedding: 查询向量, shape: (384,)
            top_k: 返回结果数
            filters: ChromaDB where 条件
            
        Returns:
            RetrievalResult 列表
        """
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
        )
        
        results = []
        for i in range(len(result["ids"][0])):
            chunk = {
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "note_id": result["metadatas"][0][i]["note_id"],
                "subject_id": result["metadatas"][0][i]["subject_id"],
                "hadm_id": result["metadatas"][0][i]["hadm_id"],
                "note_type": result["metadatas"][0][i]["note_type"],
                "chunk_index": result["metadatas"][0][i]["chunk_index"],
            }
            results.append({
                "chunk": chunk,
                "distance": result["distances"][0][i],
            })
        
        return results
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_vector_store.py -v
```

预期：PASS，2 tests passed

- [ ] **步骤 5：Commit**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: add VectorStore with ChromaDB wrapper"
```

---

## 任务 7：LLM 处理器（llm_processor.py）

**文件：**
- 创建：`src/llm_processor.py`
- 测试：`tests/test_llm_processor.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_llm_processor.py`:
```python
"""LLMProcessor 测试。"""
import pytest
import os
from unittest.mock import Mock, patch
from src.llm_processor import LLMProcessor


class TestLLMProcessor:
    @pytest.fixture
    def processor(self):
        # 使用 mock 方式测试，避免真实 API 调用
        return LLMProcessor(api_key="test-key", base_url="http://test")
    
    def test_format_chunks(self, processor):
        chunks = [
            {"chunk_id": "C1", "text": "Text one"},
            {"chunk_id": "C2", "text": "Text two"},
        ]
        formatted = processor._format_chunks(chunks)
        assert "C1" in formatted
        assert "Text one" in formatted
        assert "C2" in formatted
    
    def test_parse_rerank_response(self, processor):
        response_text = '[{"chunk_id": "C1", "score": 90, "reasoning": "relevant"}]'
        result = processor._parse_json_response(response_text)
        assert result[0]["chunk_id"] == "C1"
        assert result[0]["score"] == 90
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_llm_processor.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.llm_processor'`

- [ ] **步骤 3：编写实现代码**

`src/llm_processor.py`:
```python
"""LLM 处理器模块。

调用 kimi API 进行重排序和信息抽取。
一次只处理一个需求，分步调用。
"""
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class LLMProcessor:
    """LLM 处理器。
    
    三个独立任务：
    1. rerank: 重排序
    2. classify: 二分类
    3. extract: 信息抽取（允许 null）
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "kimi-latest",
        base_url: str = "https://api.moonshot.cn/v1",
        prompts_dir: str = "./prompts",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # 加载 Jinja2 模板
        self.env = Environment(loader=FileSystemLoader(prompts_dir))
        self.rerank_template = self.env.get_template("rerank.j2")
        self.classify_template = self.env.get_template("classify.j2")
        self.extract_template = self.env.get_template("extract.j2")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """LLM 重排序。
        
        Args:
            query: 用户查询
            chunks: 待重排序的 chunks（约 20-50 个）
            top_n: 返回前 N 个
            
        Returns:
            List[(chunk, relevance_score, reasoning)]
        """
        # 分批处理，每批 10 个
        batch_size = 10
        all_scores = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.rerank_template.render(query=query, chunks=batch)
            
            response = self._call_llm(prompt)
            scores = self._parse_json_response(response)
            
            for score_data in scores:
                chunk_id = score_data["chunk_id"]
                chunk = next((c for c in batch if c["chunk_id"] == chunk_id), None)
                if chunk:
                    all_scores.append((
                        chunk,
                        float(score_data.get("score", 0)),
                        score_data.get("reasoning", ""),
                    ))
        
        # 按分数排序
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_n]
    
    def classify(
        self,
        chunks: List[Dict[str, Any]],
        criteria: str,
    ) -> List[Tuple[Dict[str, Any], bool, str]]:
        """二分类。
        
        Args:
            chunks: 待分类的 chunks
            criteria: 分类标准
            
        Returns:
            List[(chunk, is_match, reasoning)]
        """
        batch_size = 10
        all_results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.classify_template.render(criteria=criteria, chunks=batch)
            
            response = self._call_llm(prompt)
            classifications = self._parse_json_response(response)
            
            for cls_data in classifications:
                chunk_id = cls_data["chunk_id"]
                chunk = next((c for c in batch if c["chunk_id"] == chunk_id), None)
                if chunk:
                    all_results.append((
                        chunk,
                        bool(cls_data.get("is_match", False)),
                        cls_data.get("reasoning", ""),
                    ))
        
        return all_results
    
    def extract(
        self,
        chunks: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """信息抽取。
        
        Args:
            chunks: 待抽取的 chunks
            fields: 要提取的字段列表
            
        Returns:
            List[{"chunk_id": ..., "extractions": {...}, "confidence": ...}]
            注意：字段不存在时，extractions 中对应值为 null
        """
        batch_size = 5  # 抽取任务较复杂，每批少放一些
        all_results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.extract_template.render(fields=fields, chunks=batch)
            
            response = self._call_llm(prompt)
            extractions = self._parse_json_response(response)
            
            for ext_data in extractions:
                all_results.append({
                    "chunk_id": ext_data["chunk_id"],
                    "extractions": ext_data.get("extractions", {}),
                    "confidence": ext_data.get("confidence", "medium"),
                })
        
        return all_results
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical information processing assistant. Always return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # 低温度，减少随机性
        )
        return response.choices[0].message.content
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON。"""
        # 清理可能的 markdown 代码块标记
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        return json.loads(text)
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """格式化 chunks 为文本。"""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"--- Fragment {i} ---")
            lines.append(f"ID: {chunk['chunk_id']}")
            lines.append(f"Text: {chunk['text']}")
            lines.append("")
        return "\n".join(lines)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_llm_processor.py -v
```

预期：PASS，3 tests passed（不涉及真实 API 调用）

- [ ] **步骤 5：Commit**

```bash
git add src/llm_processor.py tests/test_llm_processor.py
git commit -m "feat: add LLMProcessor with rerank/classify/extract"
```

---

## 任务 8：主流程编排（pipeline.py）

**文件：**
- 创建：`src/pipeline.py`
- 测试：`tests/test_pipeline.py`

- [ ] **步骤 1：编写失败的测试**

`tests/test_pipeline.py`:
```python
"""Pipeline 测试。"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.pipeline import NoteSearchPipeline
from src.config import PipelineConfig


class TestNoteSearchPipeline:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            persist_dir=str(tmp_path / "chroma_test"),
        )
    
    def test_pipeline_init(self, config):
        pipeline = NoteSearchPipeline(config)
        assert pipeline.config == config
    
    def test_build_index(self, config, tmp_path):
        # 创建测试数据
        test_data = pd.DataFrame({
            "note_id": ["TEST-1"],
            "subject_id": [1],
            "hadm_id": [1],
            "note_type": ["DS"],
            "note_seq": [1],
            "charttime": pd.to_datetime(["2020-01-01"]),
            "storetime": pd.to_datetime(["2020-01-01"]),
            "text": ["Patient has diabetes and hypertension."],
        })
        excel_path = tmp_path / "test.xlsx"
        test_data.to_excel(excel_path, index=False)
        
        config.excel_path = str(excel_path)
        pipeline = NoteSearchPipeline(config)
        
        # Mock chunker 和 embedder 以加速测试
        with patch.object(pipeline.chunker, 'chunk') as mock_chunk:
            mock_chunk.return_value = [
                {
                    "chunk_id": "TEST-1-0",
                    "note_id": "TEST-1",
                    "subject_id": 1,
                    "hadm_id": 1,
                    "note_type": "DS",
                    "chunk_index": 0,
                    "text": "Patient has diabetes.",
                    "token_count": 10,
                }
            ]
            
            with patch.object(pipeline.embedder, 'encode') as mock_encode:
                mock_encode.return_value = np.array([[0.1] * 384])
                
                pipeline.build_index(str(excel_path))
                
                # 验证 store 中有数据
                results = pipeline.store.query(
                    np.array([0.1] * 384),
                    top_k=10,
                )
                assert len(results) == 1
```

- [ ] **步骤 2：运行测试验证失败**

```bash
pytest tests/test_pipeline.py -v
```

预期：FAIL，`ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **步骤 3：编写实现代码**

`src/pipeline.py`:
```python
"""主流程编排模块。"""
from typing import Optional, List, Dict, Any
import numpy as np

from src.config import PipelineConfig
from src.data_loader import DataLoader
from src.semantic_chunker import SemanticChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_processor import LLMProcessor


class NoteSearchPipeline:
    """MIMIC Note 语义检索 Pipeline。
    
    当前本地脚本的核心类，未来可直接包装为 API。
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.chunker = SemanticChunker(
            embedder_model=config.model_name,
            min_chunk_tokens=config.chunk_config.min_chunk_tokens,
            target_chunk_tokens=config.chunk_config.target_chunk_tokens,
            max_chunk_tokens=config.chunk_config.max_chunk_tokens,
            similarity_percentile=config.chunk_config.similarity_percentile,
        )
        self.embedder = Embedder(config.model_name)
        self.store = VectorStore(config.persist_dir, config.collection_name)
        self.llm = LLMProcessor(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url,
            prompts_dir=config.prompts_dir,
        )
    
    def build_index(self, excel_path: str):
        """首次运行：加载 → 分块 → 向量化 → 存储。
        
        Args:
            excel_path: Excel 文件路径
        """
        print(f"Loading data from {excel_path}...")
        df = self.data_loader.load_notes(excel_path)
        print(f"Loaded {len(df)} notes.")
        
        print("Chunking notes...")
        all_chunks = []
        for _, row in df.iterrows():
            metadata = {
                "note_id": row["note_id"],
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(row["hadm_id"]),
                "note_type": row["note_type"],
                "note_seq": int(row["note_seq"]),
            }
            chunks = self.chunker.chunk(row["text"], metadata)
            all_chunks.extend(chunks)
        print(f"Generated {len(all_chunks)} chunks.")
        
        print("Encoding chunks...")
        embeddings = self.embedder.encode([c["text"] for c in all_chunks])
        print(f"Encoded {len(embeddings)} embeddings.")
        
        print("Storing in vector database...")
        self.store.add_chunks(all_chunks, embeddings)
        print("Index built successfully!")
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        enable_rerank: bool = True,
        enable_llm_extract: bool = False,
        extract_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """统一的查询入口。
        
        Args:
            query: 用户查询（自然语言）
            filters: 条件筛选，如 {"note_type": {"$eq": "DS"}}
            top_k: 返回结果数
            enable_rerank: 是否启用 LLM 重排序
            enable_llm_extract: 是否启用 LLM 信息抽取
            extract_fields: 要抽取的字段列表
            
        Returns:
            SearchResult dict
        """
        print(f"Searching for: {query}")
        
        # 1. 生成查询向量
        query_emb = self.embedder.encode([query])[0]
        
        # 2. 向量检索（扩大召回）
        retrieve_k = max(top_k, self.config.retrieve_top_k)
        results = self.store.query(query_emb, top_k=retrieve_k, filters=filters)
        print(f"Retrieved {len(results)} candidates.")
        
        # 3. LLM 重排序
        if enable_rerank and results:
            chunks = [r["chunk"] for r in results]
            reranked = self.llm.rerank(query, chunks, top_n=self.config.rerank_top_n)
            
            # 将 rerank 结果转换为标准格式
            results = [
                {
                    "chunk": chunk,
                    "distance": score / 100.0,  # 归一化到 0-1
                    "score": score,
                    "reasoning": reasoning,
                }
                for chunk, score, reasoning in reranked
            ]
            print(f"Reranked to {len(results)} results.")
        
        # 4. LLM 信息抽取
        if enable_llm_extract and results and extract_fields:
            chunks = [r["chunk"] for r in results[:top_k]]
            extractions = self.llm.extract(chunks, extract_fields)
            
            # 合并抽取结果到 results
            for i, ext in enumerate(extractions):
                if i < len(results):
                    results[i]["extractions"] = ext["extractions"]
                    results[i]["confidence"] = ext["confidence"]
            print(f"Extracted fields: {extract_fields}")
        
        return {
            "query": query,
            "filters": filters,
            "results": results[:top_k],
            "total_candidates": len(results),
        }
```

- [ ] **步骤 4：运行测试验证通过**

```bash
pytest tests/test_pipeline.py -v
```

预期：PASS，2 tests passed

- [ ] **步骤 5：Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add NoteSearchPipeline orchestrating all modules"
```

---

## 任务 9：CLI 入口（main.py）

**文件：**
- 创建：`main.py`

- [ ] **步骤 1：编写 main.py**

```python
"""CLI 入口脚本。"""
import os
import argparse
import json
from src.config import PipelineConfig
from src.pipeline import NoteSearchPipeline


def main():
    parser = argparse.ArgumentParser(description="MIMIC Note Semantic Search")
    parser.add_argument("--build-index", action="store_true", help="Build vector index")
    parser.add_argument("--excel", default="data/无标题.xlsx", help="Excel file path")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--rerank", action="store_true", help="Enable LLM rerank")
    parser.add_argument("--extract", type=str, help="Extract fields (comma-separated)")
    parser.add_argument("--filter-note-type", type=str, help="Filter by note_type")
    parser.add_argument("--api-key", default=os.getenv("KIMI_API_KEY"), help="Kimi API key")
    
    args = parser.parse_args()
    
    # 配置
    config = PipelineConfig(
        excel_path=args.excel,
        api_key=args.api_key,
    )
    
    pipeline = NoteSearchPipeline(config)
    
    if args.build_index:
        pipeline.build_index(args.excel)
    
    elif args.query:
        filters = None
        if args.filter_note_type:
            filters = {"note_type": {"$eq": args.filter_note_type}}
        
        extract_fields = None
        if args.extract:
            extract_fields = [f.strip() for f in args.extract.split(",")]
        
        result = pipeline.search(
            query=args.query,
            filters=filters,
            top_k=args.top_k,
            enable_rerank=args.rerank,
            enable_llm_extract=bool(extract_fields),
            extract_fields=extract_fields,
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：验证 CLI 可用**

```bash
python main.py --help
```

预期：显示帮助信息

- [ ] **步骤 3：Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point"
```

---

## 任务 10：集成测试与 README

**文件：**
- 创建：`README.md`

- [ ] **步骤 1：编写 README.md**

```markdown
# MIMIC Note 语义检索系统

基于 MIMIC 临床 note 数据的语义检索系统，支持自然语言查询和条件筛选。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 设置 API Key

```bash
export KIMI_API_KEY="your-api-key"
```

### 3. 构建索引

```bash
python main.py --build-index --excel data/无标题.xlsx
```

### 4. 查询

```bash
# 自然语言查询
python main.py --query "肝硬化伴腹水的治疗方案" --rerank

# 条件筛选
python main.py --query "ascites treatment" --filter-note-type DS

# 信息抽取
python main.py --query "肝硬化病例" --rerank --extract "MELD评分,利尿剂用药"
```

## 模块说明

- `src/data_loader.py`: 加载 Excel 数据
- `src/semantic_chunker.py`: 语义分块（spaCy + MiniLM）
- `src/embedder.py`: 文本向量化（all-MiniLM-L6-v2）
- `src/vector_store.py`: ChromaDB 向量存储
- `src/llm_processor.py`: LLM 重排序和信息抽取
- `src/pipeline.py`: 主流程编排

## 测试

```bash
pytest tests/ -v
```
```

- [ ] **步骤 2：运行全部测试**

```bash
pytest tests/ -v
```

预期：所有测试通过

- [ ] **步骤 3：手动集成验证**

```bash
# 构建索引
python main.py --build-index --excel "data/无标题.xlsx"

# 简单查询（不涉及 LLM）
python main.py --query "ascites" --top-k 5
```

预期：输出 JSON 格式的检索结果

- [ ] **步骤 4：Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## 自检

### 1. 规格覆盖度

| 规格需求 | 实现任务 | 状态 |
|---------|---------|------|
| 模块化 Pipeline 架构 | 任务 2-8 | ✅ |
| spaCy 分句 + MiniLM 语义分块 | 任务 5 | ✅ |
| 多信号融合（语义+结构+长度） | 任务 5 | ✅ |
| MIN=80, TARGET=250, MAX=512 | 任务 5 | ✅ |
| ChromaDB 向量存储 | 任务 6 | ✅ |
| 条件筛选（metadata过滤） | 任务 6 | ✅ |
| LLM 重排序 | 任务 7 | ✅ |
| LLM 二分类 | 任务 7 | ✅ |
| LLM 信息抽取（允许 null） | 任务 7 | ✅ |
| 一次一个需求，分步调用 | 任务 7 | ✅ |
| Web 扩展预留 | 任务 8（Pipeline 类设计） | ✅ |
| CLI 入口 | 任务 9 | ✅ |
| Prompt 模板外置 | 任务 1 | ✅ |

### 2. 占位符扫描

- ✅ 无 "TODO"、"待定"、"后续实现"
- ✅ 无 "添加适当的错误处理" 等模糊描述
- ✅ 每个代码步骤包含完整代码
- ✅ 无 "类似任务 X" 的重复引用

### 3. 类型一致性

- ✅ `PipelineConfig` 在各任务中使用一致
- ✅ `Chunk` dict 的字段名在各模块中一致
- ✅ `VectorStore.query()` 返回格式在测试和实现中一致

---

## 执行交接

**计划已完成并保存到 `docs/superpowers/plans/2026-06-10-mimic-note-retrieval-plan.md`。两种执行方式：**

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

**选哪种方式？**
