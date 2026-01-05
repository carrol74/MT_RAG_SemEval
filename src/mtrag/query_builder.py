# used when testing sample data where each task has multiple turns

from typing import Dict, Any, List


def last_user_turn(task: Dict[str, Any]) -> str:
    turns = task.get("input", [])
    for turn in reversed(turns):
        if turn.get("speaker") == "user":
            return (turn.get("text") or "").strip()
    return (turns[-1].get("text") or "").strip() if turns else ""

def concat_last_n_user_turns(task: Dict[str, Any], n: int = 3) -> str:
    turns = task.get("input", [])
    user_texts: List[str] = [
        (t.get("text") or "").strip()
        for t in turns
        if t.get("speaker") == "user" and (t.get("text") or "").strip()
    ]
    return " ".join(user_texts[-n:])
