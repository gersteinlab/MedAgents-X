import os
import json
import csv
import argparse
from collections import defaultdict

def extract_expert_profiles(logs_path):
    """Extract expert profiles from logs.json files."""
    if not os.path.exists(logs_path):
        return []
    
    with open(logs_path, 'r') as f:
        logs_data = json.load(f)
    
    experts = []
    
    for expert_key, expert_data in logs_data.items():
        if 'rounds' in expert_data:
            for round_data in expert_data['rounds']:
                if round_data.get('round_num', 0) == 0:
                    if 'expert_results' in round_data:
                        for expert_result in round_data['expert_results']:
                            if 'profile' in expert_result:
                                profile = expert_result['profile']
                                if 'Ablation' in profile.get('name', ''):
                                    continue
                                experts.append({
                                    'expert_key': expert_key,
                                    'round_num': round_data.get('round_num', 0),
                                    'name': profile.get('name', '').replace('\n', ' '),
                                    'job_title': profile.get('job_title', '').replace('\n', ' '),
                                    'past_experience': profile.get('past_experience', '').replace('\n', ' '),
                                    'educational_background': profile.get('educational_background', '').replace('\n', ' '),
                                    'research_focus': profile.get('research_focus', '').replace('\n', ' ')
                                })
    
    return experts

def discover_logs_files(output_base):
    """Discover all logs.json files in the output directory structure."""
    logs_files = []
    
    for root, dirs, files in os.walk(output_base):
        if 'medagents' not in root:
            continue
        if 'logs.json' in files:
            logs_files.append(os.path.join(root, 'logs.json'))
    
    return logs_files

def main():
    parser = argparse.ArgumentParser(description="Analyze expert profiles from MedAgents-2 logs.")
    parser.add_argument("--output_base", type=str, default="output", help="Base output directory")
    parser.add_argument("--output_csv", type=str, default="expert_profiles.csv", help="Output CSV file")
    args = parser.parse_args()

    logs_files = discover_logs_files(args.output_base)
    
    all_experts = []
    
    for logs_file in logs_files:
        experts = extract_expert_profiles(logs_file)
        
        relative_path = os.path.relpath(logs_file, args.output_base)
        path_parts = relative_path.split(os.sep)
        
        if len(path_parts) >= 4:
            dataset = path_parts[0]
            ablation = path_parts[1]
            exp_name = path_parts[2]
            run_dir = path_parts[3]
            model = path_parts[4] if len(path_parts) > 4 else 'unknown'
        else:
            dataset = 'unknown'
            ablation = 'unknown'
            exp_name = 'unknown'
            run_dir = 'unknown'
            model = 'unknown'
        
        for expert in experts:
            expert.update({
                'dataset': dataset,
                'ablation': ablation,
                'exp_name': exp_name,
                'run_dir': run_dir,
                'model': model,
                'logs_file': logs_file
            })
            all_experts.append(expert)
    
    all_experts.sort(key=lambda x: (x['dataset'], x['ablation'], x['exp_name'], x['run_dir'], x['model'], x['expert_key'], x['round_num']))
    
    output_path = os.path.join(args.output_base, args.output_csv)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'dataset', 'ablation', 'exp_name', 'run_dir', 'model',
            'expert_key', 'round_num', 'name', 'job_title', 
            'past_experience', 'educational_background', 
            'research_focus', 'logs_file'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for expert in all_experts:
            writer.writerow(expert)
    
    print(f"Expert profiles saved to {output_path}")
    print(f"Total experts processed: {len(all_experts)}")
    print(f"Total logs files processed: {len(logs_files)}")

if __name__ == "__main__":
    main()
