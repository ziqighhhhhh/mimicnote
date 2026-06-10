"""LLMProcessor 测试。"""
import pytest
from src.llm_processor import LLMProcessor


class TestLLMProcessor:
    @pytest.fixture
    def processor(self):
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
