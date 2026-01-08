"""
evaluate metrics such as RBalg, RBllm, RLF for Task B
IDK judge used for answerability correction
"""

import json
import subprocess
from typing import Dict, List
from pathlib import Path

class TaskBEvaluator:
    """Evaluate Task B predictions"""
    
    def __init__(self, eval_script_path: str = "scripts/evaluation/run_generation_eval.py"):
        """
        Args:
            eval_script_path: Path to official evaluation script
        """
        self.eval_script_path = Path(eval_script_path)
        
        if not self.eval_script_path.exists():
            print(f"Warning: Evaluation script not found at {eval_script_path}")
    
    def evaluate(
        self,
        prediction_file: str,
        output_file: str,
        provider: str = "openai",
        openai_key: str = None
    ) -> Dict:
        """
        Run official evaluation script
        
        Args:
            prediction_file: Path to predictions jsonl
            output_file: Path to save evaluation results
            provider: 'openai' or 'hf'
            openai_key: OpenAI API key if using openai provider
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Build command
        cmd = [
            "python",
            str(self.eval_script_path),
            "-i", prediction_file,
            "-o", output_file,
            "-e", "scripts/evaluation/config.yaml",
            "--provider", provider
        ]
        
        if provider == "openai" and openai_key:
            cmd.extend(["--openai_key", openai_key])
        
        # Run evaluation
        print(f"Running evaluation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr}")
            return {}
        
        # Load results
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                # The eval script appends metrics to each task
                # We need to aggregate them
                return self._aggregate_metrics(output_file)
        
        return {}
    
    def _aggregate_metrics(self, result_file: str) -> Dict:
        """
        Aggregate metrics from evaluation results
        
        Args:
            result_file: Path to evaluation results jsonl
            
        Returns:
            Dictionary of aggregated metrics
        """
        metrics_list = []
        
        with open(result_file, 'r') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    if 'metrics' in task:
                        metrics_list.append(task['metrics'])
        
        if not metrics_list:
            return {}
        
        # Aggregate
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def format_results(self, metrics: Dict) -> str:
        """
        Format evaluation results for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        lines = ["="*50]
        lines.append("Task B Evaluation Results")
        lines.append("="*50)
        
        # Main metrics
        if 'RBalg' in metrics:
            lines.append(f"RBalg:        {metrics['RBalg']:.3f}")
        if 'RBllm' in metrics:
            lines.append(f"RBllm:        {metrics['RBllm']:.3f}")
        if 'RLF' in metrics:
            lines.append(f"RLF:          {metrics['RLF']:.3f}")
        if 'ans_acc' in metrics:
            lines.append(f"Ans. Acc:     {metrics['ans_acc']:.3f}")
        
        lines.append("="*50)
        
        return "\n".join(lines)


class SimpleEvaluator:
    """
    Simplified evaluator using basic metrics
    does not rely on official evaluation script
    """
    
    def __init__(self):
        try:
            from rouge import Rouge
            from bert_score import score as bert_score
            self.rouge = Rouge()
            self.bert_score = bert_score
        except ImportError:
            print("Warning: rouge or bert_score not installed")
            print("Install with: pip install rouge-score bert-score")
            self.rouge = None
            self.bert_score = None
    
    def evaluate_simple(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """
        Simple evaluation using ROUGE and BERT score
        
        Args:
            predictions: List of predicted responses
            references: List of reference answers
            
        Returns:
            Dictionary of metrics
        """
        if not predictions or not references:
            return {}
        
        metrics = {}
        
        # ROUGE-L
        if self.rouge:
            rouge_scores = self.rouge.get_scores(predictions, references, avg=True)
            metrics['rouge_l'] = rouge_scores['rouge-l']['f']
        
        # BERT Score
        if self.bert_score:
            P, R, F1 = self.bert_score(predictions, references, lang='en', verbose=False)
            metrics['bert_score_f1'] = F1.mean().item()
            metrics['bert_score_recall'] = R.mean().item()
        
        return metrics


# Example usage
if __name__ == "__main__":
    evaluator = TaskBEvaluator()
    
    # Evaluate using official script
    metrics = evaluator.evaluate(
        prediction_file="outputs/task_b_predictions.jsonl",
        output_file="outputs/task_b_evaluation.json",
        provider="openai",
        openai_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(evaluator.format_results(metrics))

# # RBalg: harmonic mean of three algorithmic metrics
# def compute_rbalg(pred, ref, passages):
#     bert_rec = bert_recall(pred, ref)
#     bert_k_prec = bert_k_precision(pred, passages)
#     rouge_l = compute_rouge_l(pred, ref)
#     return harmonic_mean([bert_rec, bert_k_prec, rouge_l])

# # RBllm: LLM judge (需要median of 4 models)
# def compute_rbllm(pred, ref, passages, conversation):
#     judges = ['gpt-4o-mini', 'qwen-2.5-72b', 
#               'mixtral-8x22b', 'llama-3.1-405b']
#     scores = [judge(pred, ref, passages) for judge in judges]
#     return np.median(scores)

# # RLF: RAGAS Faithfulness
# def compute_rlf(pred, passages):
#     return ragas_faithfulness(pred, passages)
