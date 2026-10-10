from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .schema import Document, Turn
from .retrieval import HybridRetriever
from .memory import MemoryStore
from .pipeline import GlassesPipeline
from .data import read_jsonl, export_datasets
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="AIGT reference pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a fixed multi-turn weather fixture")
    demo.add_argument("--documents", default="data/demo/documents.jsonl")
    demo.add_argument("--output", default="outputs/demo.json")
    demo.add_argument("--base-url", help="explicit optional chat endpoint, including /v1")
    demo.add_argument("--model", help="model name when using --base-url")
    export = sub.add_parser("export", help="create grouped SFT/DPO/OPD datasets")
    export.add_argument("--input", default="data/demo/tasks.jsonl")
    export.add_argument("--output", default="outputs/datasets")
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export_datasets(read_jsonl(args.input), args.output), indent=2))
        return
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be specified together")
    generator = None
    if args.base_url:
        from .models import ChatEndpoint
        generator = ChatEndpoint(args.base_url, args.model)
    memory = MemoryStore()
    try:
        docs = [Document(**d) for d in read_jsonl(args.documents)]
        pipeline = GlassesPipeline(HybridRetriever(docs), memory, generator)
        history = [Turn("user", "明天下午去杭州，适合外出吗？", "2026-10-10T09:00:00+08:00")]
        current = Turn("user", "改成上海，简单说。", "2026-10-10T09:01:00+08:00")
        result = asdict(pipeline.ask(user_id="demo-user", scope="trip-demo",
                                    history=history, current=current))
        report = evaluate(result, {"required": ["上海", "阵雨"], "forbidden": ["杭州", "放心外出"],
                                  "city": "上海", "target_date": "2026-10-11",
                                  "evidence_ids": ["weather-sh-11"]})
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"answer": result, "evaluation": report},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"backend": result["backend"], "answer": result["text"],
                          "fixture_checks_pass": report["objective_pass"],
                          "report": str(path)}, ensure_ascii=False, indent=2))
    finally:
        memory.close()


if __name__ == "__main__":
    main()
