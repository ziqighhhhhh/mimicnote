"""向量存储/检索模块。"""
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb


class VectorStore:
    """ChromaDB 向量存储封装。"""
    
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "mimic_notes"):
        """使用内存模式 ChromaDB，完全绕过 Windows 文件锁定问题。
        
        persist_dir 参数保留以保持 API 兼容，但实际不使用。
        """
        self.collection_name = collection_name
        # 内存模式：数据存储在内存中，不涉及文件系统
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, batch_size: int = 2000):
        """分批添加 chunks，避免超过 ChromaDB 的批次限制（5461）。
        
        batch_size 设为 2000（而非 5000），避免 Windows 上 Rust 后端
        在大批量写入时的稳定性问题。
        """
        embeddings_list = embeddings.tolist()
        total = len(chunks)
        
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings_list[start:end]
            
            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [
                {
                    "note_id": c["note_id"],
                    "subject_id": c["subject_id"],
                    "hadm_id": c["hadm_id"],
                    "note_type": c["note_type"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch_chunks
            ]
            
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                print(f"  Stored batch {start}-{end} / {total}")
            except Exception as e:
                print(f"  ERROR storing batch {start}-{end}: {e}")
                raise
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
        )
        
        results = []
        for i in range(len(result["ids"][0])):
            chunk = {
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "note_id": result["metadatas"][0][i]["note_id"],
                "subject_id": result["metadatas"][0][i]["subject_id"],
                "hadm_id": result["metadatas"][0][i]["hadm_id"],
                "note_type": result["metadatas"][0][i]["note_type"],
                "chunk_index": result["metadatas"][0][i]["chunk_index"],
            }
            results.append({
                "chunk": chunk,
                "distance": result["distances"][0][i],
            })
        
        return results
