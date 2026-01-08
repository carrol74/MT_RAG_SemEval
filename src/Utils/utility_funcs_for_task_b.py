"""
Utility functions for Task B
"""

import json
from typing import Dict, List
from pathlib import Path

def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file
    
    Args:
        filepath: Path to jsonl file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """
    Save to JSONL file
    
    Args:
        data: List of dictionaries
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
