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
