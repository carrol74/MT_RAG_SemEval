import logging
from task_a.config import *
import json
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

def save_predictions(predictions, output_path):
    """
    Sample predictions format:
    {
        "conversation_id": "dd6b6ffd177f2b311abe676261279d2f",
        "task_id": "dd6b6ffd177f2b311abe676261279d2f::2",
        "Collection": "mt-rag-clapnq-elser-512-100-20240503",
        "input": [
            {
            "speaker": "user",
            "text": "where do the arizona cardinals play this week"
            }
        ]
        "contexts":
            [
                {
                    "document_id": "822086267_7384-8758-0-1374",
                    "text": "...",
                    "score": 27.759
                }, ...
            ],
    }

    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + '\n')
    logging.info(f"{len(predictions)} Predictions saved to {output_path}")