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
        results = store.query(
            embeddings[1],
            top_k=10,
            filters={"note_type": {"$eq": "DS"}}
        )
        assert len(results) == 1
        assert results[0]["chunk"]["note_type"] == "DS"
