"""Pipeline 测试。"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
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
                
                results = pipeline.store.query(
                    np.array([0.1] * 384),
                    top_k=10,
                )
                assert len(results) == 1
