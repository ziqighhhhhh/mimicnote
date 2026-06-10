"""向量化器模块。"""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """使用 all-MiniLM-L6-v2 生成文本 embedding。"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """批量生成 embedding。"""
        return self.model.encode(texts, convert_to_numpy=True)
