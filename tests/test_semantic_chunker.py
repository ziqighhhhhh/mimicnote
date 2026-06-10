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
        text = "Patient has diabetes. " * 100
        metadata = {"note_id": "TEST-2", "subject_id": 1, "hadm_id": 1,
                    "note_type": "DS", "note_seq": 1}
        chunks = chunker.chunk(text, metadata)
        
        for chunk in chunks:
            assert chunk["token_count"] <= 512
    
    def test_chunk_preserves_metadata(self, chunker):
        text = "Simple text."
        metadata = {"note_id": "TEST-3", "subject_id": 999, "hadm_id": 888,
                    "note_type": "DS", "note_seq": 42}
        chunks = chunker.chunk(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0]["note_id"] == "TEST-3"
        assert chunks[0]["subject_id"] == 999
        assert chunks[0]["hadm_id"] == 888
