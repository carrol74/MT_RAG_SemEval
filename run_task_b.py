"""
command line interface to run the full pipeline
"""

import argparse
import os
from pathlib import Path

from src.task_b.config import TaskBConfig, ModelConfig, DataConfig
from src.task_b.pipeline import TaskBPipeline

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Task B: Generation with Reference Passages")
    
    # Data arguments
    parser.add_argument(
        "--input",
        type=str,
        default="human/generation_tasks/reference.jsonl",
        help="Input file path (reference.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/task_b_predictions.jsonl",
        help="Output file path for predictions"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name (gpt-4o, gpt-4o-mini, claude-sonnet-4, llama-3.1-405b, etc.)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (1=sequential, 8=recommended)"
    )

    # API keys
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (if not in environment)"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (for Azure OpenAI, etc.)"
    )

    parser.add_argument(
        "--eval-provider",
        choices=["openai", "hf"],
        default="hf"
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Processing options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save intermediate results every N tasks"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Build configuration from arguments
    config = TaskBConfig(
        model=ModelConfig(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            api_base=args.api_base or os.getenv("OPENAI_API_BASE")
        ),
        data=DataConfig(
            input_file=args.input,
            output_file=args.output,
            eval_output=args.output.replace(".jsonl", "_eval.json")
        ),
        verbose=args.verbose,
        save_every=args.save_every
    )
    
    # Print configuration
    print("\n" + "="*80)
    print("Task B: Generation with Reference Passages")
    print("="*80)
    print(f"Model:        {config.model.model_name}")
    print(f"Temperature:  {config.model.temperature}")
    print(f"Input:        {config.data.input_file}")
    print(f"Output:       {config.data.output_file}")
    print("="*80 + "\n")
    
    # Check input file exists
    if not Path(config.data.input_file).exists():
        print(f"Error: Input file not found: {config.data.input_file}")
        return
    
    # Create and run pipeline
    try:
        pipeline = TaskBPipeline(config)
        metrics = pipeline.run()
        
        # Print final results
        print("\n" + "="*80)
        print("TASK B COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Predictions saved to: {config.data.output_file}")
        print(f"Evaluation saved to:  {config.data.eval_output}")
        print("\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running Task B: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()