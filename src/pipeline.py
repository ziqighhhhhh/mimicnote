"""主流程编排模块。"""
from typing import Optional, List, Dict, Any
import numpy as np

from src.config import PipelineConfig
from src.data_loader import DataLoader
from src.semantic_chunker import SemanticChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm_processor import LLMProcessor


class NoteSearchPipeline:
    """MIMIC Note 语义检索 Pipeline。"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.chunker = SemanticChunker(
            embedder_model=config.model_name,
            min_chunk_tokens=config.chunk_config.min_chunk_tokens,
            target_chunk_tokens=config.chunk_config.target_chunk_tokens,
            max_chunk_tokens=config.chunk_config.max_chunk_tokens,
            similarity_percentile=config.chunk_config.similarity_percentile,
        )
        self.embedder = Embedder(config.model_name)
        self.store = VectorStore(config.persist_dir, config.collection_name)
        self.llm = LLMProcessor(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url,
            prompts_dir=config.prompts_dir,
        )
    
    def build_index(self, excel_path: str):
        """首次运行：加载 → 分块 → 向量化 → 存储。"""
        print(f"Loading data from {excel_path}...")
        df = self.data_loader.load_notes(excel_path)
        print(f"Loaded {len(df)} notes.")
        
        print("Chunking notes...")
        all_chunks = []
        for _, row in df.iterrows():
            metadata = {
                "note_id": row["note_id"],
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(row["hadm_id"]),
                "note_type": row["note_type"],
                "note_seq": int(row["note_seq"]),
            }
            chunks = self.chunker.chunk(row["text"], metadata)
            all_chunks.extend(chunks)
        print(f"Generated {len(all_chunks)} chunks.")
        
        print("Encoding chunks...")
        embeddings = self.embedder.encode([c["text"] for c in all_chunks])
        print(f"Encoded {len(embeddings)} embeddings.")
        
        print("Storing in vector database...")
        self.store.add_chunks(all_chunks, embeddings)
        print("Index built successfully!")
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        enable_rerank: bool = True,
        enable_llm_extract: bool = False,
        extract_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """统一的查询入口。"""
        print(f"Searching for: {query}")
        
        query_emb = self.embedder.encode([query])[0]
        
        retrieve_k = max(top_k, self.config.retrieve_top_k)
        results = self.store.query(query_emb, top_k=retrieve_k, filters=filters)
        print(f"Retrieved {len(results)} candidates.")
        
        if enable_rerank and results:
            chunks = [r["chunk"] for r in results]
            reranked = self.llm.rerank(query, chunks, top_n=self.config.rerank_top_n)
            
            results = [
                {
                    "chunk": chunk,
                    "distance": score / 100.0,
                    "score": score,
                    "reasoning": reasoning,
                }
                for chunk, score, reasoning in reranked
            ]
            print(f"Reranked to {len(results)} results.")
        
        if enable_llm_extract and results and extract_fields:
            chunks = [r["chunk"] for r in results[:top_k]]
            extractions = self.llm.extract(chunks, extract_fields)
            
            for i, ext in enumerate(extractions):
                if i < len(results):
                    results[i]["extractions"] = ext["extractions"]
                    results[i]["confidence"] = ext["confidence"]
            print(f"Extracted fields: {extract_fields}")
        
        return {
            "query": query,
            "filters": filters,
            "results": results[:top_k],
            "total_candidates": len(results),
        }
