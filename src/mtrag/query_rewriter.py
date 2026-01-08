# BEIR MT Query Rewriter
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

def split_questions_snapshot(questions_text: str) -> Tuple[List[str], str]:
    """
    adapt 'question_text' in retrieval_tasks/<domain>_questions.jsonl

    Returns:
      history_turns (preserve original prefixes), last_question (stripped prefix)
    """
    turns = parse_questions_text_to_turns(questions_text)
    if not turns:
        return [], ""

    last = strip_speaker_prefix(turns[-1])
    history = turns[:-1]
    return history, last

def _postprocess_rewrite(s: str) -> str:
    """
    Make output safe for retrieval (search query).
    - take first non-empty line
    - strip quotes / leading labels
    - cut off any leaked prompt/instruction tail on the same line
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

    # cut off leaked instructions if they appear after the actual query
    cut_phrases = [
        "To rewrite the user's last question",
        "Rewrite the user's last question",
        "Guidelines:",
        "Output ONLY",
    ]
    for p in cut_phrases:
        idx = s.find(p)
        if idx > 0:
            s = s[:idx].strip()
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
    history_turns: int = 6


class MTQueryRewriter:
    """
    LLM-based query rewriter (query rewriting) for retrieval.

    Uses HF `pipeline(task="text-generation")` with chat messages input.
    Intended for local models that support chat templates.
    """

    def __init__(self, cfg: RewriteConfig):
        dtype = cfg.dtype
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

    def rewrite_from_questions_text(self, questions_text: str) -> str:
        """
        generate rewrite from retrieval_tasks/<domain>_questions.jsonl's text field.
        """
        history_turns, last_q = split_questions_snapshot(questions_text)
        return self.rewrite_query(query=last_q, history=history_turns)
    
    def rewrite_query(self, query: str, history: List[str]) -> str:
        """
        Args:
            query: last user question (may include '|user|:' prefix)
            history: list of previous turns; can include '|user|:'/'|assistant|:' prefixes
        """
        clean_query = strip_speaker_prefix(query)

        # Keep speaker prefixes in history to help resolution; but trim length a bit
        hist_lines = [t.strip() for t in history if t and t.strip()]
        if self.cfg.history_turns > 0 and len(hist_lines) > self.cfg.history_turns:
            hist_lines = hist_lines[-self.cfg.history_turns :]
        hist_block = "\n".join(hist_lines)

        user_prompt = (
            f"History:\n{hist_block}\n\n"
            f"Last Question:\n{clean_query}\n\n"
            "Rewritten Query:"
        )

        prompt_str = f"{self.system_prompt}\n\n{user_prompt}"

        generate_kwargs = {
            "do_sample": self.cfg.do_sample,
            "max_new_tokens": self.cfg.max_new_tokens,
            "return_full_text": False,
        }

        if self.cfg.do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
            generate_kwargs["top_p"] = self.cfg.top_p

        out = self.pipe(prompt_str, **generate_kwargs)
        gen = out[0].get("generated_text", "")

        rewritten = _postprocess_rewrite(gen)
        return rewritten if rewritten else clean_query
    