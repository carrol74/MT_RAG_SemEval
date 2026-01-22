import sys
from pathlib import Path
sys.path.insert(0, 'scripts/evaluation')

from scripts.evaluation.run_algorithmic import run_algorithmic_judges

input_file = 'outputs/task_b_21_p0.95_zeroshot_nosepa.jsonl'
p = Path(input_file)
output_file = str(p.with_name(p.stem + '_alg_metrics.jsonl')) 

run_algorithmic_judges(
    evaluator_file='scripts/evaluation/config.yaml',
    input_file=input_file,
    output_file=output_file
)

print(f"Done! Results saved to: {output_file}")

import json

tasks_with_metrics = []
with open(output_file, 'r') as f:
    for line in f:
        if line.strip():
            task = json.loads(line)
            if 'metrics' in task:
                tasks_with_metrics.append(task['metrics'])

print(f"\nFound {len(tasks_with_metrics)} tasks with metrics")

if tasks_with_metrics:
    print("\nFirst task metrics:")
    print(json.dumps(tasks_with_metrics[0], indent=2))
    
    aggregated = {}
    metric_names = tasks_with_metrics[0].keys()
    
    for metric_name in metric_names:
        values = []
        for m in tasks_with_metrics:
            if metric_name in m and m[metric_name] is not None:
                val = m[metric_name]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, list):
                    numeric = [x for x in val if isinstance(x, (int, float))]
                    if numeric:
                        values.append(sum(numeric) / len(numeric))
        
        if values:
            aggregated[metric_name] = sum(values) / len(values)
    
    print("\n" + "="*80)
    print("Aggregated Metrics:")
    print("="*80)
    for k, v in aggregated.items():
        print(f"{k:20s}: {v:.4f}")
else:
    print("No metrics found! Need to re-run evaluation.")