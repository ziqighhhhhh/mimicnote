"""语义分块器模块。

基于 spaCy 分句 + MiniLM 句向量 + 多信号融合的语义分块策略。
"""
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any


# 常见的 MIMIC note 章节标题关键词
KNOWN_SECTION_KEYWORDS = {
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HPI",
    "PAST MEDICAL HISTORY", "PMH", "MEDICATIONS", "ALLERGIES",
    "PHYSICAL EXAM", "PHYSICAL EXAMINATION", "VITALS",
    "LABORATORY DATA", "LABS", "IMAGING", "STUDIES",
    "ASSESSMENT", "PLAN", "DISPOSITION", "DISCHARGE DIAGNOSIS",
    "DISCHARGE MEDICATIONS", "FOLLOWUP", "FOLLOW UP",
    "BRIEF HOSPITAL COURSE", "HOSPITAL COURSE",
}


class SemanticChunker:
    """语义分块器。
    
    信号融合框架：
    1. 语义边界：相邻句 MiniLM cosine similarity（底部 20%）
    2. 结构边界：空行、疑似标题行
    3. 长度约束：MIN=80, TARGET=250, MAX=512 token
    """
    
    def __init__(
        self,
        embedder_model: str = "all-MiniLM-L6-v2",
        min_chunk_tokens: int = 80,
        target_chunk_tokens: int = 250,
        max_chunk_tokens: int = 512,
        similarity_percentile: int = 20,
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer(embedder_model)
        self.min_chunk_tokens = min_chunk_tokens
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.similarity_percentile = similarity_percentile
    
    def chunk(self, note_text: str, metadata: dict) -> List[Dict[str, Any]]:
        """将 note 文本切分为语义 chunk。"""
        sentences = self._split_sentences(note_text)
        if not sentences:
            return []
        
        sentence_embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        similarities = self._compute_similarities(sentence_embeddings)
        
        structural_breaks = self._detect_structural_boundaries(note_text, sentences)
        semantic_breaks = self._detect_semantic_breaks(similarities)
        final_breaks = self._merge_boundaries(semantic_breaks, structural_breaks, len(sentences))
        
        raw_chunks = self._build_raw_chunks(sentences, final_breaks)
        processed_chunks = self._apply_length_constraints(raw_chunks)
        
        return self._wrap_chunks(processed_chunks, metadata)
    
    def _split_sentences(self, text: str) -> List[str]:
        """spaCy 分句，过滤空句。"""
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) >= 5:
                sentences.append(sent_text)
        return sentences
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """计算相邻句子的 cosine similarity。"""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities
    
    def _detect_structural_boundaries(self, text: str, sentences: List[str]) -> List[int]:
        """检测结构边界（空行、标题行），返回句子索引列表。"""
        breaks = []
        lines = text.split('\n')
        
        # 构建行到句子的简化映射
        char_offset = 0
        sentence_start_positions = []
        for sent in sentences:
            pos = text.find(sent, char_offset)
            if pos >= 0:
                sentence_start_positions.append(pos)
                char_offset = pos + len(sent)
            else:
                sentence_start_positions.append(-1)
        
        # 检测空行和标题行
        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # 空行：找到空行位置之后的第一个句子
                # 计算空行在文本中的字符位置
                line_start_pos = sum(len(l) + 1 for l in lines[:line_idx])
                for sent_idx, sent_pos in enumerate(sentence_start_positions):
                    if sent_pos >= line_start_pos:
                        breaks.append(sent_idx)
                        break
            elif self._is_title_line(stripped):
                line_start_pos = sum(len(l) + 1 for l in lines[:line_idx])
                for sent_idx, sent_pos in enumerate(sentence_start_positions):
                    if sent_pos >= line_start_pos:
                        breaks.append(sent_idx)
                        break
        
        return sorted(set(b for b in breaks if b > 0 and b < len(sentences)))
    
    def _is_title_line(self, line: str) -> bool:
        """判断是否为标题行。"""
        if len(line) > 60:
            return False
        if line.isupper():
            return True
        if line.endswith(':'):
            return True
        if line in KNOWN_SECTION_KEYWORDS:
            return True
        return False
    
    def _detect_semantic_breaks(self, similarities: List[float]) -> List[int]:
        """基于百分比法检测语义边界。"""
        if not similarities:
            return []
        
        threshold = np.percentile(similarities, self.similarity_percentile)
        breaks = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        return breaks
    
    def _merge_boundaries(self, semantic_breaks: List[int], structural_breaks: List[int], 
                          num_sentences: int) -> List[int]:
        """融合语义边界和结构边界。"""
        merged = set(semantic_breaks)
        
        for sb in structural_breaks:
            nearby = any(abs(sb - sem) <= 2 for sem in semantic_breaks)
            if not nearby:
                merged.add(sb)
        
        merged = {b for b in merged if 0 < b < num_sentences}
        return sorted(merged)
    
    def _build_raw_chunks(self, sentences: List[str], breaks: List[int]) -> List[List[str]]:
        """根据边界将句子分组。"""
        if not breaks:
            return [sentences]
        
        chunks = []
        start = 0
        for break_point in breaks:
            chunks.append(sentences[start:break_point])
            start = break_point
        chunks.append(sentences[start:])
        
        return [c for c in chunks if c]
    
    def _apply_length_constraints(self, raw_chunks: List[List[str]]) -> List[List[str]]:
        """应用长度约束：合并短块，切分长块。"""
        result = []
        i = 0
        
        while i < len(raw_chunks):
            chunk = raw_chunks[i]
            token_count = self._count_tokens(chunk)
            
            if token_count < self.min_chunk_tokens:
                if i + 1 < len(raw_chunks):
                    merged = chunk + raw_chunks[i + 1]
                    merged_tokens = self._count_tokens(merged)
                    if merged_tokens <= self.max_chunk_tokens:
                        result.append(merged)
                        i += 2
                        continue
                if result and token_count < self.min_chunk_tokens:
                    prev = result[-1] + chunk
                    if self._count_tokens(prev) <= self.max_chunk_tokens:
                        result[-1] = prev
                        i += 1
                        continue
                result.append(chunk)
                i += 1
            
            elif token_count > self.max_chunk_tokens:
                sub_chunks = self._split_long_chunk(chunk)
                result.extend(sub_chunks)
                i += 1
            
            else:
                result.append(chunk)
                i += 1
        
        return result
    
    def _split_long_chunk(self, chunk: List[str]) -> List[List[str]]:
        """对超长 chunk 在局部最低相似度处二次切分。"""
        if len(chunk) <= 2:
            return [chunk]
        
        embeddings = self.embedder.encode(chunk, convert_to_numpy=True)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        
        min_idx = np.argmin(similarities)
        split_point = min_idx + 1
        
        left = chunk[:split_point]
        right = chunk[split_point:]
        
        result = []
        for sub in [left, right]:
            if self._count_tokens(sub) > self.max_chunk_tokens and len(sub) > 2:
                result.extend(self._split_long_chunk(sub))
            else:
                result.append(sub)
        
        return result
    
    def _count_tokens(self, sentences: List[str]) -> int:
        """估算 token 数（使用 spaCy token 数）。"""
        text = " ".join(sentences)
        doc = self.nlp(text)
        return len(doc)
    
    def _wrap_chunks(self, sentence_groups: List[List[str]], metadata: dict) -> List[Dict[str, Any]]:
        """将句子组包装为 Chunk dict。"""
        chunks = []
        for idx, group in enumerate(sentence_groups):
            text = " ".join(group)
            token_count = self._count_tokens(group)
            chunk = {
                "chunk_id": f"{metadata['note_id']}-{idx}",
                "note_id": metadata["note_id"],
                "subject_id": metadata["subject_id"],
                "hadm_id": metadata["hadm_id"],
                "note_type": metadata["note_type"],
                "chunk_index": idx,
                "text": text,
                "token_count": token_count,
            }
            chunks.append(chunk)
        return chunks
