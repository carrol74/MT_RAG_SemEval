import argparse
import os
from pathlib import Path
from src.task_b.config import TaskBConfig, ModelConfig, DataConfig
from src.task_c.rag_pipeline import MTRAGPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run Task C: End-to-End RAG")
    
    # Re-use Task B arguments for consistency
    parser.add_argument("--input", type=str, default="human/generation_tasks/reference.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default="outputs/task_c_predictions.jsonl", help="Output file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (Keep low for RAG to manage memory)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # config setup
    config = TaskBConfig(
        model=ModelConfig(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        data=DataConfig(
            input_file=args.input,
            output_file=args.output,
        )
    )
    
    # Add Task C specific retrieval config
    config.top_k = args.top_k

    print("\n" + "="*80)
    print("Task C: End-to-End RAG (Retrieval + Generation)")
    print("="*80)
    
    try:
        pipeline = MTRAGPipeline(config)
        pipeline.run()
        print("TASK C COMPLETED SUCCESSFULLY")
    except Exception as e:
        print(f"\nError running Task C: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()