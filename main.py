#!/usr/bin/env python3
"""
Main entry point for MedAgents-2 experiments.

This module provides a simple interface to run medical question answering experiments
using the EBMedAgents architecture. It supports both interactive mode and batch processing.
"""

import asyncio
import argparse
import json
import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI

from medagents import MedAgents
from agents import set_default_openai_client, set_tracing_disabled

# Load environment variables
load_dotenv()

def setup_openai_client():
    """Setup OpenAI client with environment variables."""
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_tracing_disabled(disabled=True)
    return client

def load_questions_from_file(file_path: str) -> List[Dict]:
    """Load questions from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_results(results: List[Dict], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def format_question_for_display(question: str, options: Dict[str, str]) -> str:
    """Format question and options for display."""
    formatted = f"\nQuestion: {question}\n\nOptions:\n"
    for key, value in options.items():
        formatted += f"  {key}: {value}\n"
    return formatted

async def run_single_experiment(
    question: str, 
    options: Dict[str, str], 
    cfg: DictConfig, 
    difficulty: Optional[str] = None
) -> Dict:
    """Run a single experiment with the given question and options."""
    print(f"\n{'='*80}")
    print(f"Running experiment with difficulty: {difficulty or 'auto'}")
    print(f"{'='*80}")
    
    # Setup OpenAI client
    setup_openai_client()
    
    # Create orchestrator
    orchestrator = MedAgents(cfg)
    
    # Run the experiment
    result = await orchestrator.run(question, options, difficulty)
    
    # Format results
    experiment_result = {
        "question": question,
        "options": options,
        "difficulty": difficulty,
        "final_answer": result.final_decision['final_answer'],
        "vote_scores": result.final_decision['vote_scores'],
        "expert_details": result.final_decision['details'],
        "rounds": len(result.rounds),
        "total_tokens": result.total_usage.total_tokens if result.total_usage else 0,
        "input_tokens": result.total_usage.input_tokens if result.total_usage else 0,
        "output_tokens": result.total_usage.output_tokens if result.total_usage else 0,
        "full_log": result.to_dict()
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS")
    print(f"{'='*80}")
    print(f"Final Answer: {experiment_result['final_answer']}")
    print(f"Vote Distribution: {experiment_result['vote_scores']}")
    print(f"Rounds: {experiment_result['rounds']}")
    print(f"Total Tokens: {experiment_result['total_tokens']:,}")
    print(f"Input Tokens: {experiment_result['input_tokens']:,}")
    print(f"Output Tokens: {experiment_result['output_tokens']:,}")
    
    return experiment_result

async def run_batch_experiments(
    questions: List[Dict], 
    cfg: DictConfig, 
    difficulty: Optional[str] = None,
    output_path: Optional[str] = None
) -> List[Dict]:
    """Run multiple experiments in batch."""
    results = []
    
    for i, q_data in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}")
        
        question = q_data['question']
        options = q_data['options']
        q_difficulty = q_data.get('difficulty', difficulty)
        
        print(format_question_for_display(question, options))
        
        try:
            result = await run_single_experiment(question, options, cfg, q_difficulty)
            results.append(result)
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            results.append({
                "question": question,
                "options": options,
                "error": str(e),
                "final_answer": None
            })
    
    # Save results if output path provided
    if output_path:
        save_results(results, output_path)
        print(f"\nResults saved to: {output_path}")
    
    return results

def interactive_mode(cfg: DictConfig):
    """Run in interactive mode where user can input questions."""
    print("\n" + "="*80)
    print("MEDAGENTS-2 INTERACTIVE MODE")
    print("="*80)
    print("Enter medical questions and options. Type 'quit' to exit.")
    print("Format: Enter question, then options A, B, C, D...")
    print("="*80)
    
    while True:
        try:
            # Get question
            print("\nEnter the medical question (or 'quit' to exit):")
            question = input("> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            # Get options
            options = {}
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
            
            print("\nEnter the answer options (press Enter twice to finish):")
            for letter in option_letters:
                option = input(f"{letter}: ").strip()
                if not option:
                    break
                options[letter] = option
            
            if not options:
                print("No options provided. Skipping...")
                continue
            
            # Get difficulty
            print("\nEnter difficulty level (easy/medium/hard) or press Enter for auto:")
            difficulty_input = input("> ").strip()
            difficulty = difficulty_input if difficulty_input in ['easy', 'medium', 'hard'] else None
            
            # Run experiment
            asyncio.run(run_single_experiment(question, options, cfg, difficulty))
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point for MedAgents-2 experiments."""
    parser = argparse.ArgumentParser(description='Run MedAgents-2 experiments')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'single'], 
                       default='interactive', help='Run mode')
    parser.add_argument('--questions', type=str, help='Path to JSON file with questions')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], 
                       help='Difficulty level (overrides auto-detection)')
    parser.add_argument('--question', type=str, help='Single question for testing')
    parser.add_argument('--options', nargs='+', help='Options for single question (A:opt1 B:opt2 ...)')
    
    parser.add_argument('--config', action='append', default=[], help='Hydra override (key=value); repeat as needed')
    args = parser.parse_args()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parent / 'conf')):
        cfg = hydra.compose(config_name='config', overrides=args.config)
    
    # Override config with command line args
    if args.difficulty:
        cfg.triage.forced_level = args.difficulty
    
    print("MedAgents-2 Configuration:")
    print(f"  Model: {cfg.execution.model.name}")
    print(f"  Context Sharing: {cfg.orchestrate.discussion_mode}")
    print(f"  Medical Specialties: {len(cfg.orchestrate.medical_specialties)}")
    print(f"  Difficulty Settings: {['easy', 'medium', 'hard']}")
    
    if args.mode == 'interactive':
        interactive_mode(cfg)
    
    elif args.mode == 'batch':
        if not args.questions:
            print("Error: --questions file required for batch mode")
            sys.exit(1)
        
        questions = load_questions_from_file(args.questions)
        print(f"Loaded {len(questions)} questions from {args.questions}")
        
        results = asyncio.run(run_batch_experiments(
            questions, cfg, args.difficulty, args.output
        ))
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Total Questions: {len(results)}")
        print(f"Successful: {len([r for r in results if 'error' not in r])}")
        print(f"Failed: {len([r for r in results if 'error' in r])}")
        
        if results:
            total_tokens = sum(r.get('total_tokens', 0) for r in results)
            print(f"Total Tokens Used: {total_tokens:,}")
    
    elif args.mode == 'single':
        if not args.question or not args.options:
            print("Error: --question and --options required for single mode")
            print("Example: --question 'What is...' --options 'A:Option1' 'B:Option2'")
            sys.exit(1)
        
        # Parse options
        options = {}
        for opt in args.options:
            if ':' in opt:
                key, value = opt.split(':', 1)
                options[key.strip().upper()] = value.strip()
            else:
                print(f"Warning: Invalid option format: {opt}")
        
        if not options:
            print("Error: No valid options provided")
            sys.exit(1)
        
        print(format_question_for_display(args.question, options))
        
        result = asyncio.run(run_single_experiment(
            args.question, options, cfg, args.difficulty
        ))
        
        if args.output:
            save_results([result], args.output)
            print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main() 