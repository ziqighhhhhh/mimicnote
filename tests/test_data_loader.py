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
