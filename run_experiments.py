#!/usr/bin/env python3
"""
MedAgents-2 Experiment Runner (main entry)
"""
import os
import json
from typing import List, Dict, Any
from datetime import datetime
import asyncio
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from medagents import MedAgents
from main import setup_openai_client
from experiment import ExperimentResult, ExperimentSaver

load_dotenv()

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

async def process_query(problem: Dict[str, Any], cfg: DictConfig) -> ExperimentResult:
    """Process a single query using the new agent implementation."""
    start_time = datetime.now()
    setup_openai_client()
    orchestrator = MedAgents(cfg)
    difficulty = None if not cfg.triage.disable_triage else cfg.triage.forced_level
    medagents_log = await orchestrator.run(problem['question'], problem['options'], difficulty)
    return ExperimentResult.from_medagents_log(
        realidx=problem.get('realidx', 0),
        question=problem['question'],
        options=problem['options'],
        answer_idx=problem.get('answer_idx'),
        difficulty=difficulty or 'auto',
        time_taken=(datetime.now() - start_time).total_seconds(),
        medagents_log=medagents_log
    )

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    """Main entry point for MedAgents-2 experiments."""
    task_name = cfg.execution.dataset.name
    experiment_name = cfg.execution.experiment_name
    run_id = cfg.execution.experiments.run_id
    model_name = cfg.execution.model.name.replace('-', '_')
    output_base = cfg.execution.output.folder
    output_dir = os.path.join(output_base, task_name, experiment_name, f"run_{run_id}", model_name)
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, 'w') as f:
        OmegaConf.save(cfg, f)
    saver = ExperimentSaver(output_dir)
    problems = load_jsonl(os.path.join(cfg.execution.dataset.dir, cfg.execution.dataset.name, f"{cfg.execution.dataset.split}.jsonl"))
    for idx, problem in enumerate(problems):
        if 'realidx' not in problem:
            problem['realidx'] = idx
    processed_realidx = saver.get_processed_realidx()
    problems_to_process = [problem for problem in problems if problem['realidx'] not in processed_realidx]
    print(f"Processing {len(problems_to_process)} problems out of {len(problems)} total problems.")
    print(f"Output directory: {output_dir}")
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Retrieval device: {cfg.search.hardware.device}")
    async def process_all():
        for idx, problem in enumerate(problems_to_process):
            print(f"Processing problem {idx+1}/{len(problems_to_process)} (realidx: {problem['realidx']})")
            try:
                result = await process_query(problem, cfg)
                saver.add_result(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing problem {problem['realidx']}: {e}")
    asyncio.run(process_all())
    print(f"Experiment completed. Total results: {len(saver.results)}")

if __name__ == "__main__":
    main() 