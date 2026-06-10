"""LLM 处理器模块。

调用 kimi API 进行重排序和信息抽取。
一次只处理一个需求，分步调用。
"""
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI


class LLMProcessor:
    """LLM 处理器。
    
    三个独立任务：
    1. rerank: 重排序
    2. classify: 二分类
    3. extract: 信息抽取（允许 null）
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "kimi-latest",
        base_url: str = "https://api.moonshot.cn/v1",
        prompts_dir: str = "./prompts",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # 加载 Jinja2 模板
        self.env = Environment(loader=FileSystemLoader(prompts_dir))
        self.rerank_template = self.env.get_template("rerank.j2")
        self.classify_template = self.env.get_template("classify.j2")
        self.extract_template = self.env.get_template("extract.j2")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """LLM 重排序。"""
        batch_size = 10
        all_scores = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.rerank_template.render(query=query, chunks=batch)
            
            response = self._call_llm(prompt)
            scores = self._parse_json_response(response)
            
            for score_data in scores:
                chunk_id = score_data["chunk_id"]
                chunk = next((c for c in batch if c["chunk_id"] == chunk_id), None)
                if chunk:
                    all_scores.append((
                        chunk,
                        float(score_data.get("score", 0)),
                        score_data.get("reasoning", ""),
                    ))
        
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_n]
    
    def classify(
        self,
        chunks: List[Dict[str, Any]],
        criteria: str,
    ) -> List[Tuple[Dict[str, Any], bool, str]]:
        """二分类。"""
        batch_size = 10
        all_results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.classify_template.render(criteria=criteria, chunks=batch)
            
            response = self._call_llm(prompt)
            classifications = self._parse_json_response(response)
            
            for cls_data in classifications:
                chunk_id = cls_data["chunk_id"]
                chunk = next((c for c in batch if c["chunk_id"] == chunk_id), None)
                if chunk:
                    all_results.append((
                        chunk,
                        bool(cls_data.get("is_match", False)),
                        cls_data.get("reasoning", ""),
                    ))
        
        return all_results
    
    def extract(
        self,
        chunks: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """信息抽取。字段不存在时返回 null。"""
        batch_size = 5
        all_results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            prompt = self.extract_template.render(fields=fields, chunks=batch)
            
            response = self._call_llm(prompt)
            extractions = self._parse_json_response(response)
            
            for ext_data in extractions:
                all_results.append({
                    "chunk_id": ext_data["chunk_id"],
                    "extractions": ext_data.get("extractions", {}),
                    "confidence": ext_data.get("confidence", "medium"),
                })
        
        return all_results
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical information processing assistant. Always return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON。"""
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        return json.loads(text)
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """格式化 chunks 为文本。"""
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"--- Fragment {i} ---")
            lines.append(f"ID: {chunk['chunk_id']}")
            lines.append(f"Text: {chunk['text']}")
            lines.append("")
        return "\n".join(lines)
