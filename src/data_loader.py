"""数据加载器模块。"""
import pandas as pd


class DataLoader:
    """加载 MIMIC note 数据。"""
    
    def load_notes(self, excel_path: str) -> pd.DataFrame:
        """读取 Excel，返回标准化 DataFrame。
        
        过滤掉 text 为空的记录。
        """
        df = pd.read_excel(excel_path)
        
        # 确保必要列存在
        required_columns = ["note_id", "subject_id", "hadm_id", "note_type", 
                           "note_seq", "charttime", "storetime", "text"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # 过滤空文本
        df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        return df
