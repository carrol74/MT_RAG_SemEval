"""
Evaluator for Task B
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Optional

class TaskBEvaluator:
    """
    Evaluate Task B predictions using official MTRAGEval script
    """
    
    def __init__(self, 
                 eval_script_dir: str = "scripts/evaluation"):
        """
        Args:
            eval_script_dir: Path to official evaluation scripts
        """
        self.eval_script_dir = Path(eval_script_dir)
        
        # set all necessary path attributes
        self.gen_eval_script = self.eval_script_dir / "run_generation_eval.py"
        self.format_checker = self.eval_script_dir / "format_checker.py"
        self.config_file = self.eval_script_dir / "config.yaml" 
        
        # Check if scripts exist
        if not self.gen_eval_script.exists():
            raise FileNotFoundError(
                f"Official evaluation script not found: {self.gen_eval_script}\n"
                f"Please ensure you have the mt-rag-benchmark repository with scripts/evaluation/"
            )
        
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_file}\n"
                f"Please ensure scripts/evaluation/config.yaml exists"
            )
        
        # check and create __init__.py files if missing
        self._ensure_init_files()
    
    def _ensure_init_files(self):
        """Ensure __init__.py files exist for module imports"""
        scripts_init = Path("scripts") / "__init__.py"
        eval_init = self.eval_script_dir / "__init__.py"
        
        for init_file in [scripts_init, eval_init]:
            if not init_file.exists():
                print(f"Creating {init_file}...")
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.touch()
    
    def check_format(self, 
                    input_file: str, 
                    prediction_file: str) -> bool:
        """Check prediction file format using official format checker"""
        print("\nChecking prediction file format...")
        
        # Use -m to run as module
        cmd = [
            "python", "-m",
            "scripts.evaluation.format_checker",
            "--input_file", input_file,
            "--prediction_file", prediction_file,
            "--mode", "generation_taskb"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Format check failed:\n{result.stderr}")
            return False
        
        if "Format is valid" in result.stdout:
            print("✓ Format check passed!")
            return True
        else:
            print("✗ Format check failed!")
            return False
    
    def evaluate(self,
                prediction_file: Optional[str] = None,
                output_file: Optional[str] = None,
                provider: str = "hf",
                judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
                openai_key: Optional[str] = None,
                azure_host: Optional[str] = None
                ) -> Dict:
        
        if prediction_file is None or output_file is None:
            raise ValueError("prediction_file and output_file must be provided")

        # Build command - use -m to run as module
        cmd = [
            "python", "-m",
            "scripts.evaluation.run_generation_eval",
            "-i", prediction_file,
            "-o", output_file,
            "-e", str(self.config_file),
            "--provider", provider
        ]
        
        if provider == "hf":
            cmd.extend(["--judge_model", judge_model])
        elif provider == "openai":
            cmd.extend(["--openai_key", openai_key])
        """
        Run official MTRAGEval evaluation
        
        Args:
            prediction_file: Path to predictions jsonl
            output_file: Path to save evaluation results
            provider: 'openai' or 'hf'
            openai_key: OpenAI API key (if provider='openai')
            azure_host: Azure endpoint (if using Azure OpenAI)
            judge_model: HuggingFace model name (if provider='hf')
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("Running Official MTRAGEval Evaluation")
        print("="*80)
    
        
        # Add provider-specific arguments
        if provider == "openai":
            if openai_key:
                cmd.extend(["--openai_key", openai_key])
            if azure_host:
                cmd.extend(["--azure_host", azure_host])
        elif provider == "hf":
            if judge_model:
                cmd.extend(["--judge_model", judge_model])
            else:
                cmd.extend(["--judge_model", "Qwen/Qwen2.5-7B-Instruct"])
        
        print(f"Command: {' '.join(cmd)}")
        print("\nThis may take a while (especially for large datasets)...")
        print()
        
        # Run evaluation with proper working directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Check result
        if result.returncode != 0:
            print(f"Evaluation failed!")
            print(f"Error output:\n{result.stderr}")
            
            # Try fallback with PYTHONPATH
            print("\nRetrying with PYTHONPATH set...")
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                env=env
            )
            
            if result.returncode != 0:
                print(f"Still failed after retry!")
                print(f"Error output:\n{result.stderr}")
                return {}
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Load and parse results
        if Path(output_file).exists():
            metrics = self._parse_results(output_file)
            return metrics
        else:
            print(f"Warning: Output file not found: {output_file}")
            return {}
    
    def _parse_results(self, result_file: str) -> Dict:
        """
        Parse evaluation results from output file
        
        The official script outputs one JSON per line with metrics added
        We aggregate these metrics
        """
        tasks_with_metrics = []
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        if 'metrics' in task:
                            tasks_with_metrics.append(task['metrics'])
        except Exception as e:
            print(f"Error parsing results: {e}")
            return {}
        
        if not tasks_with_metrics:
            print("Warning: No metrics found in results")
            return {}
        
        # Aggregate metrics across all tasks
        aggregated = {}
        
        # Get all metric names from first task
        metric_names = tasks_with_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [
                m[metric_name] 
                for m in tasks_with_metrics 
                if metric_name in m and m[metric_name] is not None
            ]
            if values:
                aggregated[metric_name] = sum(values) / len(values)
        
        return aggregated
    
    def format_results(self, metrics: Dict) -> str:
        """
        Format evaluation results for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        if not metrics:
            return "No metrics available"
        
        lines = []
        lines.append("="*80)
        lines.append("Official MTRAGEval Results")
        lines.append("="*80)
        
        # Main metrics (in order of importance)
        main_metrics = ['RBalg', 'RBllm', 'RLF', 'ans_acc']
        
        for key in main_metrics:
            if key in metrics:
                lines.append(f"{key:15} {metrics[key]:.4f}")
        
        # Other metrics
        other_metrics = {
            k: v for k, v in metrics.items() 
            if k not in main_metrics
        }
        
        if other_metrics:
            lines.append("-"*80)
            for key, value in sorted(other_metrics.items()):
                lines.append(f"{key:15} {value:.4f}")
        
        lines.append("="*80)
        
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    import os
    
    print("Testing TaskBEvaluator...")
    
    try:
        # Initialize evaluator
        evaluator = TaskBEvaluator()
        
        print(f"\nEvaluator initialized successfully!")
        print(f"Evaluation script: {evaluator.gen_eval_script}")
        print(f"Config file: {evaluator.config_file}")
        print(f"Config exists: {evaluator.config_file.exists()}")
        
        # Check if prediction file exists
        pred_file = "outputs/task_b_predictions.jsonl"
        if not Path(pred_file).exists():
            print(f"\nPrediction file not found: {pred_file}")
            print("Please run the generation pipeline first:")
            print("  python -m src.task_b.run_task_b")
            exit(1)
        
        # Run evaluation
        print(f"\n{'='*80}")
        print("Starting evaluation...")
        print(f"{'='*80}\n")
        
        metrics = evaluator.evaluate(
            prediction_file=pred_file,
            output_file="outputs/task_b_evaluation.json",
            provider="hf",
            openai_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Display results
        if metrics:
            print("\n" + evaluator.format_results(metrics))
        else:
            print("\nNo metrics returned from evaluation")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("1. You are in the project root directory")
        print("2. scripts/evaluation/run_generation_eval.py exists")
        print("3. scripts/evaluation/config.yaml exists")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()