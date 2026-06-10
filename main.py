"""CLI 入口脚本。"""
import os
import argparse
import json
from dotenv import load_dotenv
from src.config import PipelineConfig
from src.pipeline import NoteSearchPipeline

# 加载 .env 文件（如果存在）
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="MIMIC Note Semantic Search")
    parser.add_argument("--build-index", action="store_true", help="Build vector index")
    parser.add_argument("--excel", default="data/无标题.xlsx", help="Excel file path")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--rerank", action="store_true", help="Enable LLM rerank")
    parser.add_argument("--extract", type=str, help="Extract fields (comma-separated)")
    parser.add_argument("--filter-note-type", type=str, help="Filter by note_type")
    parser.add_argument("--api-key", default=os.getenv("KIMI_API_KEY"), help="Kimi API key")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        excel_path=args.excel,
        api_key=args.api_key,
    )
    
    pipeline = NoteSearchPipeline(config)
    
    if args.build_index:
        pipeline.build_index(args.excel)
    
    elif args.query:
        filters = None
        if args.filter_note_type:
            filters = {"note_type": {"$eq": args.filter_note_type}}
        
        extract_fields = None
        if args.extract:
            extract_fields = [f.strip() for f in args.extract.split(",")]
        
        result = pipeline.search(
            query=args.query,
            filters=filters,
            top_k=args.top_k,
            enable_rerank=args.rerank,
            enable_llm_extract=bool(extract_fields),
            extract_fields=extract_fields,
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
