import json
from pathlib import Path
from typing import Dict


def load_beir_queries(queries_jsonl: Path) -> Dict[str, str]:
    """
    BEIR queries jsonl: each line like {"_id": "...", "text": "..."} or {"id": "...", "text": "..."}.
    Returns: {qid: query_text}
    """
    queries: Dict[str, str] = {}
    with queries_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("_id", None) or obj.get("id", None)
            if qid is None:
                raise KeyError(f"Missing _id/id in query line: {obj}")
            queries[str(qid)] = (obj.get("text") or "").strip()
    return queries

def load_beir_qrels_tsv(qrels_tsv: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")

            # skip header
            if parts[0].lower() == "query-id":
                continue

            if len(parts) < 3:
                raise ValueError(f"Bad qrels line: {line}")

            qid, docid, rel = parts[0], parts[1], parts[2]
            qrels.setdefault(qid, {})[docid] = int(float(rel))
    return qrels
