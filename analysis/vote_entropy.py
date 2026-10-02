#!/usr/bin/env python3
"""
Analyze vote entropy across discussion rounds to measure consensus convergence.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import glob

def calculate_entropy(vote_scores):
    """Calculate entropy from vote scores dictionary."""
    if not vote_scores:
        return 0.0
    
    # Get probability distribution from vote scores
    total = sum(vote_scores.values())
    if total == 0:
        return 0.0
    
    probs = [score / total for score in vote_scores.values() if score > 0]
    
    # Calculate Shannon entropy: H = -sum(p * log2(p))
    if len(probs) <= 1:
        return 0.0  # Perfect consensus
    
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropy

def extract_vote_data_from_logs(logs_file):
    """Extract vote data from a logs.json file."""
    try:
        with open(logs_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {logs_file}: {e}")
        return []
    
    vote_data = []
    
    for question_id, question_data in data.items():
        # Extract final decision vote scores and determine actual rounds completed
        final_vote_scores = None
        final_entropy = None
        num_rounds_completed = 0
        
        if 'final_decision' in question_data and 'vote_scores' in question_data['final_decision']:
            final_vote_scores = question_data['final_decision']['vote_scores']
            final_entropy = calculate_entropy(final_vote_scores)
            num_rounds_completed = len(question_data.get('rounds', []))
        
        # Extract round-by-round data and fill missing rounds
        round_entropies = {}  # round_num -> entropy
        
        if 'rounds' in question_data:
            rounds = question_data['rounds']
            
            for round_info in rounds:
                round_num = round_info.get('round_num', 0)
                
                # Check expert results for individual expert voting patterns
                if 'expert_results' in round_info:
                    # Calculate vote distribution from expert answers
                    expert_votes = {}
                    for expert in round_info['expert_results']:
                        if 'response' in expert and 'answer' in expert['response']:
                            answer = expert['response']['answer']
                            expert_votes[answer] = expert_votes.get(answer, 0) + 1
                    
                    if expert_votes:
                        # Convert counts to proportions for entropy calculation
                        total_votes = sum(expert_votes.values())
                        vote_proportions = {k: v/total_votes for k, v in expert_votes.items()}
                        entropy = calculate_entropy(vote_proportions)
                        round_entropies[round_num] = entropy
                        
                        vote_data.append({
                            'question_id': question_id,
                            'round': round_num,
                            'entropy': entropy,
                            'vote_scores': vote_proportions,
                            'vote_type': 'expert_distribution'
                        })
        
        # Fill missing rounds (0, 1, 2) with carried-forward values
        max_round = 2  # We want data for rounds 0, 1, 2
        for target_round in range(max_round + 1):
            if target_round not in round_entropies:
                # Find the last available round's entropy to carry forward
                carry_forward_entropy = None
                carry_forward_scores = None
                
                # Look for the highest available round <= target_round
                for r in range(target_round, -1, -1):
                    if r in round_entropies:
                        # Find the corresponding vote data
                        for entry in vote_data:
                            if (entry['question_id'] == question_id and 
                                entry['round'] == r and 
                                entry['vote_type'] == 'expert_distribution'):
                                carry_forward_entropy = entry['entropy']
                                carry_forward_scores = entry['vote_scores']
                                break
                        break
                
                # If no previous round found, use final decision if available
                if carry_forward_entropy is None and final_entropy is not None:
                    carry_forward_entropy = final_entropy
                    carry_forward_scores = final_vote_scores
                
                # Add carried-forward entry
                if carry_forward_entropy is not None:
                    vote_data.append({
                        'question_id': question_id,
                        'round': target_round,
                        'entropy': carry_forward_entropy,
                        'vote_scores': carry_forward_scores,
                        'vote_type': 'expert_distribution_carried_forward'
                    })
    
    return vote_data

def analyze_discussion_modes():
    """Analyze all discussion modes and calculate entropy convergence."""
    base_dir = Path("output/medqa/discussion_mode_ablation")
    
    all_entropy_data = []
    
    discussion_modes = ['group_chat_with_orchestrator', 'group_chat_voting_only', 
                       'independent', 'one_on_one_sync']
    
    for mode in discussion_modes:
        mode_dir = base_dir / mode
        if not mode_dir.exists():
            continue
            
        print(f"Processing {mode}...")
        
        # Find all logs.json files for this mode
        logs_files = glob.glob(str(mode_dir / "**/logs.json"), recursive=True)
        
        for logs_file in logs_files:
            # Extract run and model info from path
            path_parts = Path(logs_file).parts
            run_id = None
            model = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('run_'):
                    run_id = part
                    if i + 1 < len(path_parts):
                        model = path_parts[i + 1]
                    break
            
            vote_data = extract_vote_data_from_logs(logs_file)
            
            for entry in vote_data:
                entry['discussion_mode'] = mode
                entry['run_id'] = run_id
                entry['model'] = model
                all_entropy_data.append(entry)
    
    return pd.DataFrame(all_entropy_data)

def main():
    """Main analysis function."""
    print("Analyzing vote entropy across discussion rounds...")
    
    # Extract all entropy data
    df = analyze_discussion_modes()
    
    if df.empty:
        print("No vote data found!")
        return
    
    print(f"Found {len(df)} vote records across all experiments")
    print(f"Discussion modes: {df['discussion_mode'].unique()}")
    print(f"Rounds found: {sorted(df['round'].unique())}")
    
    # Calculate average entropy by discussion mode and round
    entropy_summary = df.groupby(['discussion_mode', 'round'])['entropy'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Save detailed data
    df.to_csv('vote_entropy_detailed.csv', index=False)
    entropy_summary.to_csv('vote_entropy_summary.csv', index=False)
    
    print("\nEntropy Summary by Discussion Mode and Round:")
    print(entropy_summary)
    
    # Calculate convergence metrics (entropy decrease over rounds)
    convergence_data = []
    
    for mode in df['discussion_mode'].unique():
        mode_data = df[df['discussion_mode'] == mode]
        
        # Group by question and calculate entropy change
        for question_id in mode_data['question_id'].unique():
            question_data = mode_data[mode_data['question_id'] == question_id]
            rounds_sorted = question_data.sort_values('round')
            
            if len(rounds_sorted) > 1:
                initial_entropy = rounds_sorted.iloc[0]['entropy']
                final_entropy = rounds_sorted.iloc[-1]['entropy']
                entropy_change = initial_entropy - final_entropy
                
                convergence_data.append({
                    'discussion_mode': mode,
                    'question_id': question_id,
                    'initial_entropy': initial_entropy,
                    'final_entropy': final_entropy,
                    'entropy_change': entropy_change,
                    'num_rounds': len(rounds_sorted)
                })
    
    convergence_df = pd.DataFrame(convergence_data)
    convergence_df.to_csv('vote_convergence_analysis.csv', index=False)
    
    print(f"\nSaved files:")
    print(f"- vote_entropy_detailed.csv: {len(df)} detailed records")
    print(f"- vote_entropy_summary.csv: {len(entropy_summary)} summary records")
    print(f"- vote_convergence_analysis.csv: {len(convergence_df)} convergence records")
    
    # Print convergence summary
    if not convergence_df.empty:
        print("\nConvergence Summary (Average entropy change by mode):")
        convergence_summary = convergence_df.groupby('discussion_mode')['entropy_change'].agg([
            'mean', 'std', 'count'
        ]).round(3)
        print(convergence_summary)

if __name__ == "__main__":
    main()