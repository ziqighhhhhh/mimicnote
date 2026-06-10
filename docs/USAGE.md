# MIMIC Note 语义检索系统 - 使用指南

## 快速开始

### 1. 环境准备

确保已安装所有依赖：

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 配置 API Key

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 Kimi API Key：

```
KIMI_API_KEY=your-api-key-here
```

或者在命令行直接设置：

```bash
# Windows PowerShell
$env:KIMI_API_KEY="your-api-key-here"

# Windows CMD
set KIMI_API_KEY=your-api-key-here

# Linux/Mac
export KIMI_API_KEY="your-api-key-here"
```

### 3. 构建索引（首次运行必需）

```bash
python main.py --build-index --excel "data/无标题.xlsx"
```

构建过程：
1. 加载 Excel 数据
2. 语义分块（spaCy + MiniLM）
3. 生成向量嵌入
4. 存入 ChromaDB（持久化到 `./chroma_db/`）

**注意**：首次构建需要下载 all-MiniLM-L6-v2 模型，可能需要几分钟。

### 4. 查询

#### 4.1 简单向量检索

```bash
python main.py --query "ascites treatment" --top-k 5
```

#### 4.2 启用 LLM 重排序

```bash
python main.py --query "肝硬化伴腹水的治疗方案" --rerank
```

#### 4.3 条件筛选

```bash
python main.py --query "ascites treatment" --filter-note-type DS
```

#### 4.4 信息抽取

```bash
python main.py --query "肝硬化病例" --rerank --extract "MELD评分,利尿剂用药"
```

#### 4.5 组合使用

```bash
python main.py --query "肝硬化失代偿" \
  --filter-note-type DS \
  --rerank \
  --extract "MELD评分,Child-Pugh分级,腹水治疗方案" \
  --top-k 10
```

---

## CLI 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--build-index` | 构建向量索引 | `--build-index` |
| `--excel` | Excel 文件路径 | `--excel data/无标题.xlsx` |
| `--query` | 查询文本 | `--query "肝硬化腹水"` |
| `--top-k` | 返回结果数量 | `--top-k 10` |
| `--rerank` | 启用 LLM 重排序 | `--rerank` |
| `--extract` | 抽取字段（逗号分隔）| `--extract "MELD评分,用药"` |
| `--filter-note-type` | 按 note_type 筛选 | `--filter-note-type DS` |
| `--api-key` | Kimi API Key | `--api-key sk-xxx` |

---

## 系统架构

```
用户查询 → Embedder(向量化) → VectorStore(检索Top-50) → LLM(重排序Top-10) → LLM(信息抽取) → 结果
                ↑                                              ↑
         ChromaDB 索引                                    kimi API
```

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据加载 | `src/data_loader.py` | 读取 Excel，过滤空文本 |
| 语义分块 | `src/semantic_chunker.py` | spaCy分句 + MiniLM语义边界 + 长度约束 |
| 向量化 | `src/embedder.py` | all-MiniLM-L6-v2 生成 384 维向量 |
| 向量存储 | `src/vector_store.py` | ChromaDB 持久化存储与检索 |
| LLM处理 | `src/llm_processor.py` | rerank / classify / extract |
| 主流程 | `src/pipeline.py` | 编排以上模块 |

---

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个模块测试
pytest tests/test_semantic_chunker.py -v
pytest tests/test_embedder.py -v
```

---

## 常见问题

**Q: 首次运行很慢？**
A: 首次需要下载 all-MiniLM-L6-v2 模型（约 80MB）和 spacy 英文模型，之后会从缓存加载。

**Q: ChromaDB 索引存在哪里？**
A: 默认保存在 `./chroma_db/` 目录下，已被 `.gitignore` 忽略。

**Q: 如何更换向量模型？**
A: 修改 `src/config.py` 中的 `model_name`，或使用兼容 SentenceTransformer 的模型。

**Q: 支持其他 LLM 吗？**
A: 当前使用 OpenAI SDK 兼容的 kimi API。如需更换其他模型，修改 `src/llm_processor.py` 中的 `base_url` 和 `model`。
