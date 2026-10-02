import os
import json
import csv
import numpy as np
from collections import defaultdict
import argparse
import glob


# Historical rates used by the preserved experiments; not a current price list.
PRICING = {
    'gpt_4o_mini': (0.15, 0.6),
    'gpt_4o': (2.5, 10),
    'o3_mini': (1.1, 4.4),
    'o1_mini': (1.1, 4.4),
    'claude_3_5_sonnet': (3.0, 15.0),
    'claude_3_5_haiku': (0.8, 4.0),
    'QwQ_32B': (1.2, 1.2),
    'DeepSeek_R1': (7, 7),
    'DeepSeek_V3': (1.25, 1.25),
    'Llama_3.3_70B_Instruct_Turbo': (0.88, 0.88)
}


def calculate_cost_from_token_usage(data, model):
    if model in PRICING:
        prompt_rate, completion_rate = PRICING[model]
        return (data['avg_input_tokens'] * prompt_rate + 
                data['avg_output_tokens'] * completion_rate) / 1000000

def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_usage(usage_path):
    if not os.path.exists(usage_path):
        return None
    with open(usage_path, 'r') as f:
        return json.load(f)

def calculate_error_bar(values):
    if len(values) <= 1:
        return 0.0
    return np.std(values, ddof=1) / np.sqrt(len(values))

def discover_experiments(output_base):
    experiments = defaultdict(lambda: defaultdict(set))
    
    for dataset in os.listdir(output_base):
        dataset_path = os.path.join(output_base, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        for ablation in os.listdir(dataset_path):
            ablation_path = os.path.join(dataset_path, ablation)
            if not os.path.isdir(ablation_path):
                continue
                
            for exp_name in os.listdir(ablation_path):
                exp_path = os.path.join(ablation_path, exp_name)
                if not os.path.isdir(exp_path):
                    continue
                    
                experiments[dataset][ablation].add(exp_name)
    
    return experiments

def save_individual_run_results(results, output_base):
    individual_results = []
    
    for dataset, ablations in results.items():
        for ablation, exp_names in ablations.items():
            for exp_name, models in exp_names.items():
                for model, runs in models.items():
                    for run_id in range(3):
                        if run_id < len(runs):
                            run = runs[run_id]
                            individual_results.append({
                                'dataset': dataset,
                                'ablation': ablation,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': f"{run['accuracy'] * 100:.1f}",
                                'avg_time': f"{run['avg_time']:.2f}",
                                'avg_cost': f"{run['avg_cost']:.4f}",
                                'count': int(run['count'])
                            })
                        else:
                            individual_results.append({
                                'dataset': dataset,
                                'ablation': ablation,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': '',
                                'avg_time': '',
                                'avg_cost': '',
                                'count': ''
                            })
    
    individual_results.sort(key=lambda x: (x['ablation'], x['exp_name'], x['model'], x['run_id']))
    
    with open(os.path.join(output_base, 'individual_runs.csv'), 'w', newline='') as csvfile:
        fieldnames = ['ablation', 'exp_name', 'dataset', 'model', 'run_id', 'accuracy', 'avg_time', 'avg_cost', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in individual_results:
            writer.writerow(result)

def save_ablation_tables(results, output_base):
    ablation_summaries = defaultdict(list)
    
    for dataset, ablations in results.items():
        for ablation, exp_names in ablations.items():
            for exp_name, models in exp_names.items():
                for model, runs in models.items():
                    for run_id in range(3):
                        if run_id < len(runs):
                            run = runs[run_id]
                            ablation_summaries[ablation].append({
                                'dataset': dataset,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': f"{run['accuracy'] * 100:.1f}",
                                'avg_time': f"{run['avg_time']:.2f}",
                                'avg_cost': f"{run['avg_cost']:.4f}",
                                'count': int(run['count'])
                            })
                        else:
                            ablation_summaries[ablation].append({
                                'dataset': dataset,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': '',
                                'avg_time': '',
                                'avg_cost': '',
                                'count': ''
                            })
    
    for ablation, data in ablation_summaries.items():
        data.sort(key=lambda x: (x['exp_name'], x['dataset'], x['model'], x['run_id']))
        
        filename = os.path.join(output_base, f'{ablation}_summary.csv')
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['exp_name', 'dataset', 'model', 'run_id', 'accuracy', 'avg_time', 'avg_cost', 'count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in data:
                writer.writerow(result)

def main():
    parser = argparse.ArgumentParser(description="Analyze MedAgents-2 experiment results.")
    parser.add_argument("--output_base", type=str, default="output", help="Base output directory")
    parser.add_argument("--output_csv", type=str, default="results_summary.csv", help="Output CSV file")
    args = parser.parse_args()

    experiments = discover_experiments(args.output_base)
    
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    summary_results = []
    
    for dataset, ablations in experiments.items():
        for ablation, exp_names in ablations.items():
            for exp_name in exp_names:
                exp_path = os.path.join(args.output_base, dataset, ablation, exp_name)
                
                if not os.path.isdir(exp_path):
                    continue
                
                runs_data = defaultdict(list)
                
                for run_dir in os.listdir(exp_path):
                    if not run_dir.startswith('run_'):
                        continue
                        
                    run_path = os.path.join(exp_path, run_dir)
                    if not os.path.isdir(run_path):
                        continue
                    
                    for model_dir in os.listdir(run_path):
                        model_path = os.path.join(run_path, model_dir)
                        if not os.path.isdir(model_path):
                            continue
                        
                        metrics_path = os.path.join(model_path, "metrics.json")
                        
                        metrics = load_metrics(metrics_path)
                        
                        if metrics is None:
                            continue
                        
                        acc = metrics['accuracy_metrics']
                        time = metrics['time_metrics']
                        
                        accuracy = acc['overall_accuracy']
                        avg_time = time['average_time']
                        count = acc['total_answers']
                        
                        avg_cost = 0.0
                        avg_cost = calculate_cost_from_token_usage(metrics['token_usage_metrics']['average_usage'], model_dir)
                        
                        run_data = {
                            'accuracy': accuracy,
                            'avg_time': avg_time,
                            'avg_cost': avg_cost,
                            'count': count
                        }
                        
                        runs_data[model_dir].append(run_data)
                        all_results[dataset][ablation][exp_name][model_dir].append(run_data)
                
                for model, runs in runs_data.items():
                    for run_id in range(3):
                        if run_id < len(runs):
                            run = runs[run_id]
                            summary_results.append({
                                'dataset': dataset,
                                'ablation': ablation,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': run['accuracy'] * 100,
                                'avg_time': run['avg_time'],
                                'avg_cost': run['avg_cost'],
                                'count': run['count']
                            })
                        else:
                            summary_results.append({
                                'dataset': dataset,
                                'ablation': ablation,
                                'exp_name': exp_name,
                                'model': model,
                                'run_id': run_id,
                                'accuracy': '',
                                'avg_time': '',
                                'avg_cost': '',
                                'count': ''
                            })
    
    summary_results.sort(key=lambda x: (x['ablation'], x['exp_name'], x['model'], x['run_id']))
    
    formatted_results = []
    for result in summary_results:
        if result['accuracy'] == '':
            formatted_result = {
                'ablation': result['ablation'],
                'exp_name': result['exp_name'],
                'dataset': result['dataset'],
                'model': result['model'],
                'run_id': result['run_id'],
                'accuracy': '',
                'avg_time': '',
                'avg_cost': '',
                'count': ''
            }
        else:
            formatted_result = {
                'ablation': result['ablation'],
                'exp_name': result['exp_name'],
                'dataset': result['dataset'],
                'model': result['model'],
                'run_id': result['run_id'],
                'accuracy': f"{result['accuracy']:.1f}",
                'avg_time': f"{result['avg_time']:.2f}",
                'avg_cost': f"{result['avg_cost']:.4f}",
                'count': int(result['count'])
            }
        formatted_results.append(formatted_result)
    
    with open(os.path.join(args.output_base, args.output_csv), 'w', newline='') as csvfile:
        fieldnames = ['ablation', 'exp_name', 'dataset', 'model', 'run_id', 'accuracy', 
                     'avg_time', 'avg_cost', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in formatted_results:
            writer.writerow(result)
    
    save_individual_run_results(all_results, args.output_base)
    save_ablation_tables(all_results, args.output_base)
    
    print(f"Results saved to {os.path.join(args.output_base, args.output_csv)}")
    print(f"Individual run results saved to {os.path.join(args.output_base, 'individual_runs.csv')}")
    print(f"Ablation-specific tables saved to {args.output_base}/*_summary.csv files")
    print(f"Total experiments processed: {len(summary_results)}")

if __name__ == "__main__":
    main()