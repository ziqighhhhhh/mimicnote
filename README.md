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
