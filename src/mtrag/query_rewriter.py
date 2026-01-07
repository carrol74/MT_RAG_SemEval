from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import pipeline


_SPEAKER_PREFIX_RE = re.compile(r"^\s*\|(user|assistant)\|\s*:\s*", re.IGNORECASE)


def strip_speaker_prefix(s: str) -> str:
    """Remove leading '|user|:' / '|assistant|:' if present."""
    return _SPEAKER_PREFIX_RE.sub("", (s or "").strip()).strip()


def parse_questions_text_to_turns(questions_text: str) -> List[str]:
    """
    Parse a BEIR-style 'questions' text field, typically like:
      '|user|: ...\n|assistant|: ...\n|user|: ...'
    Returns a list of lines (turns) with prefixes preserved.
    """
    if not questions_text:
        return []
    lines = [ln.strip() for ln in questions_text.splitlines() if ln.strip()]
    return lines


def extract_history_from_questions_text(questions_text: str, lastturn_text: str) -> List[str]:
    """
    Build history turns (excluding the last user question).
    We rely on the 'questions' file containing the whole conversation,
    and 'lastturn' containing only the final user query.

    Strategy:
    - Parse turns from questions_text
    - Remove the last turn if it matches lastturn_text (after normalization)
    - Return remaining turns as history
    """
    turns = parse_questions_text_to_turns(questions_text)
    if not turns:
        return []

    norm_last = strip_speaker_prefix(lastturn_text).lower()
    # Try dropping the final line if it matches the lastturn
    if turns:
        last_line_norm = strip_speaker_prefix(turns[-1]).lower()
        if norm_last and last_line_norm == norm_last:
            turns = turns[:-1]

    return turns


def _postprocess_rewrite(s: str) -> str:
    """
    Make output safe for retrieval (search query).
    - take first non-empty line
    - strip quotes / leading labels
    """
    if s is None:
        return ""
    s = s.strip()

    # Some models may output a label like "Rewritten Query:"
    s = re.sub(r"^\s*(rewritten\s*query\s*[:\-]\s*)", "", s, flags=re.IGNORECASE).strip()

    # take first non-empty line
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break

    # strip wrapping quotes
    s = s.strip().strip('"').strip("'").strip()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class RewriteConfig:
    model_name: str
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    dtype: Optional[torch.dtype] = None


class MTQueryRewriter:
    """
    LLM-based query rewriter (query rewriting) for retrieval.

    Uses HF `pipeline(task="text-generation")` with chat messages input.
    Intended for local models that support chat templates.
    """

    def __init__(self, cfg: RewriteConfig):
        dtype = cfg.dtype
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.cfg = cfg
        self.pipe = pipeline(
            task="text-generation",
            model=cfg.model_name,
            device_map="auto",
            dtype=dtype,
        )

        self.system_prompt = (
            "You are a Query Rewriting Module for Information Retrieval.\n\n"
            "Rewrite the user's last question into a concise, standalone search query.\n\n"
            "Guidelines:\n"
            "1. Resolve references using the conversation history (pronouns, ellipsis).\n"
            "2. Preserve and explicitly name key entities (people, places, events, products).\n"
            "3. Remove conversational or polite language.\n"
            "4. Do NOT answer the question.\n"
            "5. Do NOT add new information.\n"
            "6. Prefer keyword-rich phrasing suitable for retrieval.\n"
            "7. Keep it concise (one sentence or phrase).\n\n"
            "Output ONLY the rewritten query text."
        )

    def rewrite_query(self, query: str, history: List[str]) -> str:
        """
        Args:
            query: last user question (may include '|user|:' prefix)
            history: list of previous turns; can include '|user|:'/'|assistant|:' prefixes
        """
        clean_query = strip_speaker_prefix(query)

        # Keep speaker prefixes in history to help resolution; but trim length a bit
        hist_lines = [t.strip() for t in history if t and t.strip()]
        hist_lines = hist_lines[-6:]
        hist_block = "\n".join(hist_lines)

        user_prompt = (
            f"History:\n{hist_block}\n\n"
            f"Last Question:\n{clean_query}\n\n"
            "Rewritten Query:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        out = self.pipe(
            messages,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            return_full_text=False,
        )

        # Robustly parse common HF outputs
        gen = out[0].get("generated_text")

        if isinstance(gen, str):
            return _postprocess_rewrite(gen)

        # Some chat pipelines return list[dict(role, content)]
        if isinstance(gen, list) and gen and isinstance(gen[-1], dict):
            return _postprocess_rewrite(gen[-1].get("content", ""))

        return _postprocess_rewrite(str(gen))
