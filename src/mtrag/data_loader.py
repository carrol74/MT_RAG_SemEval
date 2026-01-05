import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL rows to avoid loading everything at once."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

@dataclass
class Corpus:
    doc_ids: List[str]
    texts: List[str]
    titles: List[str]

def load_corpus_jsonl(jsonl_path: Path) -> Corpus:
    """
    Load a passage-level corpus from a .jsonl file.

    Expected fields per line (common in this repo's corpora):
      - id (required)
      - text (optional but usually present)
      - title (optional)
    """
    doc_ids, texts, titles = [], [], []
    for obj in iter_jsonl(jsonl_path):
        if "id" not in obj:
            raise KeyError(f"Missing 'id' field in corpus row from {jsonl_path}")
        doc_ids.append(obj["id"])
        texts.append(obj.get("text", "") or "")
        titles.append(obj.get("title", "") or "")
    return Corpus(doc_ids=doc_ids, texts=texts, titles=titles)

def load_tasks_jsonl(input_jsonl: Path) -> List[Dict[str, Any]]:
    """
    Load Task A/C style input JSONL:
      - conversation_id
      - task_id
      - Collection
      - input: list[{speaker, text}]
    """
    return list(iter_jsonl(input_jsonl))
