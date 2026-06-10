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
