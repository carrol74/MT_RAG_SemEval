"""
integrated data loading, generation, evaluation for Task B
"""

from tqdm import tqdm
from typing import Dict, List

from src.task_b.config import TaskBConfig
from src.task_b.data_loader import TaskBDataLoader
from src.task_b.prompt_builder import PromptBuilder
from src.task_b.generator import GeneratorFactory
from src.task_b.evaluator import TaskBEvaluator
from Utils.utility_funcs_for_task_b import save_jsonl

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
        Generate responses for all tasks
        
        Args:
            tasks: List of tasks
            
        Returns:
            List of tasks with predictions added
        """
        results = []
        
        for i, task in enumerate(tqdm(tasks, desc="Generating")):
            # Parse task
            parsed = self.data_loader.parse_task(task)
            
            # Build prompt
            prompt = self.prompt_builder.build_prompt(
                passages=parsed['passages'],
                conversation=parsed['conversation']
            )
            
            # Generate response
            try:
                response = self.generator.generate(prompt)
            except Exception as e:
                print(f"\nError generating for task {parsed['task_id']}: {e}")
                response = "Error generating response"
            
            # Add prediction to task
            result = task.copy()
            result['predictions'] = [{'text': response}]
            results.append(result)
            
            # Verbose output
            if self.config.verbose and i < 3:  # Show first 3
                print(f"\n{'='*80}")
                print(f"Task {parsed['task_id']}")
                print(f"Question: {parsed['conversation'][-1]['text']}")
                print(f"Response: {response[:200]}...")
                print(f"{'='*80}\n")
            
            # Save intermediate results
            if (i + 1) % self.config.save_every == 0:
                temp_file = f"{self.config.data.output_file}.temp"
                save_jsonl(results, temp_file)
                print(f"Saved intermediate results to {temp_file}")
        
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
        
        save_jsonl(predictions, filepath)
        print(f"Saved {len(predictions)} predictions to {filepath}")
    
    def _evaluate(self) -> Dict:
        """
        Evaluate predictions using official script
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.evaluator.evaluate(
            prediction_file=self.config.data.output_file,
            output_file=self.config.data.eval_output,
            provider="openai",
            openai_key=self.config.model.api_key
        )
        
        # Display results
        print("\n" + self.evaluator.format_results(metrics))
        
        return metrics
    
    def generate_single(self, task: Dict) -> str:
        """
        Generate response for a single task (for debugging)
        
        Args:
            task: Single task dictionary
            
        Returns:
            Generated response
        """
        parsed = self.data_loader.parse_task(task)
        
        prompt = self.prompt_builder.build_prompt(
            passages=parsed['passages'],
            conversation=parsed['conversation']
        )
        
        return self.generator.generate(prompt)


# Example usage
if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    # Create pipeline
    pipeline = TaskBPipeline(DEFAULT_CONFIG)
    
    # Run complete pipeline
    metrics = pipeline.run()
    
    print("\nFinal Results:")
    print(f"RBalg: {metrics.get('RBalg', 'N/A'):.3f}")
    print(f"RBllm: {metrics.get('RBllm', 'N/A'):.3f}")
    print(f"RLF: {metrics.get('RLF', 'N/A'):.3f}")