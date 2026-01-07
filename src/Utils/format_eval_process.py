import os
from src.config import *
import json
from typing import List, Dict, Any
from enum import Enum
class TaskType(Enum):
    TaskTypeA = "TaskA"
    TaskTypeB = "TaskB"
    TaskTypeC = "TaskC"
    TaskTypeT = "TaskTest"

def process_data(input_file_path, task_type):
    """
    Reads the MTRAG benchmark file and converts it into a list of evaluation tasks.
    Each task consists of the history up to that point and the current user query.
    """
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                    record = json.loads(line)
                    if record.get("input")[-1].get("speaker") != "user":
                        continue  # Skip if the last input is not from user
                    # Filter columns based on task type
                    if task_type == TaskType.TaskTypeA or task_type == TaskType.TaskTypeC:
                        # Input for Task A & C: conversation_id, task_id, Collection, input
                        filtered_record = {
                            "conversation_id": record.get("conversation_id"),
                            "task_id": record.get("task_id"),
                            "Collection": record.get("Collection"),
                            "input": record.get("input")
                        }

                    elif task_type == TaskType.TaskTypeB:
                        # Input for Task B: conversation_id, task_id, Collection, input, contexts
                        filtered_record = {
                            "conversation_id": record.get("conversation_id"),
                            "task_id": record.get("task_id"),
                            "Collection": record.get("Collection"),
                            "input": record.get("input"),
                            "contexts": record.get("contexts", [])
                        }
                    else:  # TaskTypeT: all columns
                        filtered_record = record
                    
                    data.append(filtered_record)

    return data