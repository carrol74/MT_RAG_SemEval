"""
load reference.jsonl and parse data from conversation and passages
"""

import json
from typing import List, Dict, Any
from pathlib import Path

from Utils.utility_funcs_for_task_b import load_jsonl

class TaskBDataLoader:
    """Load and parse Task B data"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: Path to reference.jsonl
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Load all tasks from jsonl file
        
        Returns:
            List of task dictionaries
        """
        tasks = load_jsonl(str(self.file_path))
        
        print(f"Loaded {len(tasks)} tasks from {self.file_path}")
        return tasks
    
    def parse_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a single task into structured format
        
        Args:
            task: Raw task dictionary
            
        Returns:
            Parsed task with extracted fields
        """
        return {
            # === Input for Generator (will be fed to model) ===
            'conversation': task['input'],     # Feed to generator
            'passages': task['contexts'],      # Feed to generator
            
            # === Metadata (for evaluation only, NOT fed to generator) ===
            'reference_answer': task.get('targets', [{}])[0].get('text', ''),  # target/gold answer, for evaluation comparison
            'answerability': task.get('answerability', ['ANSWERABLE'])[0],     # For IDK correction & analysis
            'question_type': task.get('Question Type', []),                     # For analysis
            'multi_turn_type': task.get('Multi-Turn', []),                      # For analysis
            
            # === Task identifiers ===
            'task_id': task['task_id'],
            'conversation_id': task['conversation_id'],
            'turn': task.get('turn', -1),
            'collection': task.get('Collection', ''),
        }
    
    def get_conversation_text(self, conversation: List[Dict]) -> tuple:
        """
        Extract conversation history and current question
        
        Args:
            conversation: List of turns
            
        Returns:
            (history_turns, current_question)
        """
        if not conversation:
            return [], ""
        
        # All turns except the last one
        history = conversation[:-1]
        
        # Current question (last user turn)
        current_turn = conversation[-1]
        current_question = current_turn['text']
        
        return history, current_question
    
    def get_passages_text(self, passages: List[Dict]) -> List[str]:
        """
        Extract passage texts
        
        Args:
            passages: List of passage dictionaries
            
        Returns:
            List of passage text strings
        """
        return [p.get('text', '') for p in passages]


# Example usage
if __name__ == "__main__":
    loader = TaskBDataLoader("human/generation_tasks/reference.jsonl")
    tasks = loader.load_tasks()
    
    # Parse first task
    if tasks:
        parsed = loader.parse_task(tasks[0])
        print(f"\nTask ID: {parsed['task_id']}")
        print(f"Turn: {parsed['turn']}")
        print(f"Answerability: {parsed['answerability']}")
        
        history, question = loader.get_conversation_text(parsed['conversation'])
        print(f"\nHistory turns: {len(history)}")
        print(f"Current question: {question}")
        print(f"Number of passages: {len(parsed['passages'])}")