# metrics parser

import json

input_file = "outputs/task_b_evaluation.jsonl"

tasks_with_metrics = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            task = json.loads(line)
            if 'metrics' in task:
                tasks_with_metrics.append(task['metrics'])

print(f"Found {len(tasks_with_metrics)} tasks with metrics")

if tasks_with_metrics:
    print("\n" + "="*80)
    print("Available metrics:")
    print("="*80)
    for k, v in tasks_with_metrics[0].items():
        value_type = type(v).__name__
        if isinstance(v, list):
            value_sample = f"[{v[0]:.4f}...]" if v else "[]"
        else:
            value_sample = f"{v:.4f}"
        print(f"{k:30s}: {value_type:10s} {value_sample}")
    
    print("\n" + "="*80)
    print("Aggregated Metrics:")
    print("="*80)
    
    aggregated = {}
    for metric_name in tasks_with_metrics[0].keys():
        values = []
        for m in tasks_with_metrics:
            if metric_name in m and m[metric_name] is not None:
                val = m[metric_name]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, list) and val:
                    numeric_vals = [x for x in val if isinstance(x, (int, float)) and x is not None]
                    if numeric_vals:
                        values.append(sum(numeric_vals) / len(numeric_vals))
        
        if values:
            aggregated[metric_name] = sum(values) / len(values)
    
    print("\nAlgorithmic Metrics:")
    alg_metrics = ['Recall', 'RougeL_stemFalse', 'BertscoreP', 'BertscoreR', 'RB_agg']
    for k in alg_metrics:
        if k in aggregated:
            print(f"  {k:30s}: {aggregated[k]:.4f}")
    
    print("\nLLM Judge Metrics:")
    llm_metrics = ['RL_F', 'RB_llm', 'RL_F_idk', 'RB_llm_idk', 'RB_agg_idk']
    for k in llm_metrics:
        if k in aggregated:
            print(f"  {k:30s}: {aggregated[k]:.4f}")
    
    print("\nOther Metrics:")
    other_metrics = ['Length', 'idk_eval', 'BertKPrec', 'Extractiveness_RougeL']
    for k in other_metrics:
        if k in aggregated:
            print(f"  {k:30s}: {aggregated[k]:.4f}")