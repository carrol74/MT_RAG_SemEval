"""
integrated data loading, generation, evaluation for Task B
"""

from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

from src.task_b.config import TaskBConfig
from src.task_b.data_loader import TaskBDataLoader
from src.task_b.prompt_builder import PromptBuilder
from src.task_b.generator import GeneratorFactory
from src.task_b.evaluator import TaskBEvaluator
from src.Utils.utility_funcs_for_task_b import save_jsonl


class TaskBPipeline:
    """Complete Task B pipeline"""
    
    def __init__(self, config: TaskBConfig):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Task B configuration
        """
        self.config = config
        
        # Initialize components
        print("Initializing Task B pipeline...")
        
        # Data loader
        self.data_loader = TaskBDataLoader(config.data.input_file)
        
        # Prompt builder
        self.prompt_builder = PromptBuilder(config.prompt)
        
        # Generator
        self.generator = GeneratorFactory.create_generator(config.model)
        
        # Evaluator
        self.evaluator = TaskBEvaluator()
        
        print("Pipeline initialized successfully")
    
    def run(self) -> Dict:
        """
        Run complete Task B pipeline
        
        Returns:
            Dictionary of evaluation metrics
        """
        # 1. Load tasks
        print("\n" + "="*80)
        print("Step 1: Loading tasks")
        print("="*80)
        tasks = self.data_loader.load_tasks()
        
        # 2. Generate responses
        print("\n" + "="*80)
        print("Step 2: Generating responses")
        print("="*80)
        predictions = self._generate_all(tasks)
        
        # 3. Save predictions
        print("\n" + "="*80)
        print("Step 3: Saving predictions")
        print("="*80)
        self._save_predictions(predictions)
        
        # 4. Evaluate
        print("\n" + "="*80)
        print("Step 4: Evaluating")
        print("="*80)
        metrics = self._evaluate()
        
        return metrics
    
    def _generate_all(self, tasks: List[Dict]) -> List[Dict]:
        """
        Generate responses for all tasks with batch processing
        """
        print(f"\nGenerating responses for {len(tasks)} tasks...")
        print(f"Using batch size: {self.config.batch_size}")
        
        # Build all prompts first
        all_prompts = []
        for task in tasks:
            parsed = self.data_loader.parse_task(task)
            prompt = self.prompt_builder.build_prompt(
                passages=parsed['passages'],
                conversation=parsed['conversation']
            )
            all_prompts.append(prompt)
        
        # Batch generation
        try:
            responses = self.generator.generate_batch(
                all_prompts,
                batch_size=self.config.batch_size
            )
        except Exception as e:
            print(f"Batch generation failed: {e}")
            print("Falling back to sequential generation...")
            # Fallback to sequential
            responses = []
            for prompt in tqdm(all_prompts, desc="Generating (sequential)"):
                try:
                    response = self.generator.generate(prompt)
                    responses.append(response)
                except Exception as e:
                    print(f"Error: {e}")
                    responses.append("Error generating response")
        
        # Add predictions to tasks
        results = []
        for i, (task, response) in enumerate(zip(tasks, responses)):
            result = task.copy()
            result['predictions'] = [{'text': response}]
            results.append(result)
            
            # Verbose output (first 3 examples)
            if self.config.verbose and i < 3:
                parsed = self.data_loader.parse_task(task)
                print(f"\n{'='*80}")
                print(f"Example {i+1}")
                print(f"Question: {parsed['conversation'][-1]['text'][:100]}...")
                print(f"Response: {response[:200]}...")
                print(f"{'='*80}\n")
        
        return results
    
    def _save_predictions(self, predictions: List[Dict], filepath: str = None):
        """
        Save predictions to jsonl file
        
        Args:
            predictions: List of predictions
            filepath: Output file path (optional)
        """
        if filepath is None:
            filepath = self.config.data.output_file
        
        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_jsonl(predictions, filepath)
        print(f"Saved {len(predictions)} predictions to {filepath}")
    
    def _evaluate(self) -> Dict:
        """
        Evaluate predictions using official script
        
        Returns:
            Dictionary of metrics
        """
        if not Path(self.config.data.output_file).exists():
            print("Predictions file not found, skipping evaluation")
            return {}
        try:
            # choose evaluation provider (hf or openai)
            if self.config.eval_provider == "hf":
                metrics = self.evaluator.evaluate(
                    prediction_file=self.config.data.output_file,
                    output_file=self.config.data.eval_output,
                    provider="hf",
                    judge_model=self.config.model.judge_model
                )
            elif self.config.eval_provider == "openai":
                metrics = self.evaluator.evaluate(
                    prediction_file=self.config.data.output_file,
                    output_file=self.config.data.eval_output,
                    provider="openai",
                    openai_key=self.config.model.api_key
                )
            else:
                raise ValueError(f"Unknown eval_provider: {self.config.eval_provider}")
            
            if metrics:
                print("\n" + self.evaluator.format_results(metrics))
            else:
                print("\nEvaluation produced no metrics")
            
            return metrics
            
        except Exception as e:
            print(f"\nEvaluation failed: {e}")
            print("Predictions were saved, but evaluation could not be completed")
            return {}


# Example usage
if __name__ == "__main__":
    from src.task_b.config import TaskBConfig
    
    # Create default config
    config = TaskBConfig()
    
    # Create pipeline
    pipeline = TaskBPipeline(config)
    
    # Run complete pipeline
    metrics = pipeline.run()
    
    # Print results
    print("\nFinal Results:")
    if metrics:
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")
    else:
        print("No metrics available")