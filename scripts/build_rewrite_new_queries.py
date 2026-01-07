import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from src.mtrag.query_rewriter import (
    MTQueryRewriter,
    RewriteConfig,
    extract_history_from_questions_text,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_cache(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_cache(cache_path: Path, cache: Dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_maps(questions_items: List[Dict[str, Any]], lastturn_items: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    q_questions = {}
    q_lastturn = {}

    for obj in questions_items:
        qid = obj.get("_id") or obj.get("id")
        if qid is None:
            raise KeyError(f"Missing _id/id in questions jsonl line: {obj}")
        q_questions[str(qid)] = (obj.get("text") or "")

    for obj in lastturn_items:
        qid = obj.get("_id") or obj.get("id")
        if qid is None:
            raise KeyError(f"Missing _id/id in lastturn jsonl line: {obj}")
        q_lastturn[str(qid)] = (obj.get("text") or "")

    return q_questions, q_lastturn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["clapnq", "cloud", "fiqa", "govt"])
    ap.add_argument("--model_name", required=True, help="HF model name/path for chat LLM")
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--questions_jsonl", type=str, default=None)
    ap.add_argument("--lastturn_jsonl", type=str, default=None)
    ap.add_argument("--out_jsonl", type=str, default=None)

    ap.add_argument("--cache_json", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Debug: only process first N queries (0 means all).")
    ap.add_argument("--force", action="store_true", help="Ignore cache and rewrite everything.")
    args = ap.parse_args()

    domain = args.domain

    questions_path = Path(args.questions_jsonl) if args.questions_jsonl else Path(f"human/retrieval_tasks/{domain}/{domain}_questions.jsonl")
    lastturn_path = Path(args.lastturn_jsonl) if args.lastturn_jsonl else Path(f"human/retrieval_tasks/{domain}/{domain}_lastturn.jsonl")
    out_path = Path(args.out_jsonl) if args.out_jsonl else Path(f"corpora/rewrite_query/{domain}/{domain}_rewrite_new.jsonl")

    cache_path = Path(args.cache_json) if args.cache_json else Path(f"runs/rewrite_cache/{domain}_rewrite_new_cache.json")

    if not questions_path.exists():
        raise FileNotFoundError(f"Missing: {questions_path}")
    if not lastturn_path.exists():
        raise FileNotFoundError(f"Missing: {lastturn_path}")

    questions_items = read_jsonl(questions_path)
    lastturn_items = read_jsonl(lastturn_path)

    q_questions, q_lastturn = build_maps(questions_items, lastturn_items)

    cfg = RewriteConfig(model_name="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=args.max_new_tokens)
    rewriter = MTQueryRewriter(cfg)

    cache = {} if args.force else load_cache(cache_path)

    out_items: List[Dict[str, Any]] = []
    qids = list(q_lastturn.keys())
    if args.limit and args.limit > 0:
        qids = qids[: args.limit]

    for qid in tqdm(qids, desc=f"rewrite_new({domain})"):
        last_q = q_lastturn[qid]
        full_conv = q_questions.get(qid, "")

        if (not args.force) and (qid in cache):
            rewritten = cache[qid]
        else:
            history = extract_history_from_questions_text(full_conv, last_q)
            rewritten = rewriter.rewrite_query(query=last_q, history=history)

            # hard safety: never empty
            if not rewritten.strip():
                rewritten = last_q

            cache[qid] = rewritten

        out_items.append({"_id": qid, "text": rewritten})

    write_jsonl(out_path, out_items)
    save_cache(cache_path, cache)

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] cache: {cache_path}")


if __name__ == "__main__":
    main()
