
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from omegaconf import DictConfig
from natsort import natsorted

from agents import Agent, Runner, ModelSettings, handoff, Usage
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# ——————————————————————————————————————————————
# Configuration dataclass for prompts & model
# ——————————————————————————————————————————————
class ExpertProfile(BaseModel):
    name: str = Field(..., description="Name of the expert")
    past_experience: str = Field(..., description="Detailed summary of clinical or research experience")
    educational_background: str = Field(..., description="Academic degrees, supervisors, certifications, and specialized training")
    core_specialties: str = Field(..., description="Key areas of expertise with precise descriptions")

class TriageOutput(BaseModel):
    difficulty: str = Field(
        ..., 
        description="Question difficulty level", 
        pattern="^(easy|medium|hard|custom)$"
    )
    justification: str = Field(..., description="Reasoning for the assigned difficulty level")
    specialties: List[str] = Field(..., description="List of selected medical specialties")
    weights: List[float] = Field(..., description="Weights for each specialty (must sum to 1)")
    job_titles: List[str] = Field(..., description="Unique job titles for each expert")
    expert_profiles: List[ExpertProfile] = Field(..., description="Detailed expert profiles for each specialty")
    research_focuses: List[str] = Field(..., description="Unique research focus areas for each expert")
 
@dataclass
class TriageConfig:
    difficulty_system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical education specialist and triage coordinator."
    difficulty_template: str = (
        "You need to complete the following steps:\n\n"
        "STEP 1: Assess the difficulty level of the following medical question:\n\n"
        "{question}\n\n"
        "Please evaluate the question based on these criteria:\n"
        "1. Knowledge depth required (basic facts vs. specialized knowledge)\n"
        "2. Clinical reasoning complexity (straightforward vs. multi-step reasoning)\n"
        "3. Ambiguity level (clear vs. nuanced distinctions between options)\n"
        "4. Specialized terminology or concepts\n"
        "5. Rarity of the medical condition or treatment discussed\n\n"
        "Classify the question as 'easy', 'medium', or 'hard' and provide a brief justification.\n\n"
        "STEP 2: After determining the difficulty level, you MUST transfer to the appropriate triage agent to get the specialty classification and expert profiles. Do not provide the final triage output yourself - you must handoff to the specialized triage agent."
    )
    triage_system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical question triage expert."
    triage_template: str = (
        "You need to complete the following steps:\n"
        "1. Read the scenario carefully:\n'''{question}'''.\n"
        "2. Classify the question into {n} experts selected from: {fields}.\n"
        "3. For each selected expert, assign a weight between 0 and 1 (weights must sum to 1).\n"
        "4. Additionally, for each expert, generate a creative, CV-style expert profile along with a unique job title. The profile should be written like a professional curriculum vitae and include (each line should be a separate paragraph):\n"
        "   - Name: The name of the expert.\n"
        "   - Past Experience: Provide a detailed summary of clinical or research experience (e.g., 'BS in Biomedical Sciences from Stanford with 5 years in advanced clinical research').\n"
        "   - Educational Background: List academic degrees, supervisors, certifications, and specialized training with specifics such as institution names and honors (e.g., 'MD from Harvard, Fellowship in Cardiology at Mayo Clinic').\n"
        "   - Core Specialties: Enumerate key areas of expertise with precise and creative descriptions.\n"
        "5. For each expert, also provide a unique research focus that distinguishes them from other experts in the same field. This should be a specific area they've published on or have special interest in.\n"
    )

@dataclass
class TriageRunResult:
    """Result from running a triage agent."""
    response: TriageOutput
    usage: Usage

_default_cfg = TriageConfig()

def get_medical_specialties(cfg: DictConfig) -> List[str]:
    specialties = cfg.get('orchestrate', {}).get('medical_specialties', [])
    return natsorted(set(specialties))

# ——————————————————————————————————————————————
# Triage agent runner function
# ——————————————————————————————————————————————
async def run_triage_agent(question: str, cfg: DictConfig):
    if getattr(cfg.triage, 'disable_triage', False):
        forced_level = getattr(cfg.triage, 'forced_level', 'easy')
        if forced_level == 'custom':
            forced_cfg = cfg.triage.forced_level_custom
        else:
            forced_cfg = cfg.triage[forced_level]
        n = forced_cfg['num_experts']
        specialties = get_medical_specialties(cfg)[:n]
        weights = [1.0 / n] * n
        job_titles = [f"Ablation Expert {i+1}" for i in range(n)]
        expert_profiles = [
            dict(
                name=f"Ablation Expert {i+1}",
                past_experience="Ablation study: profile not generated.",
                educational_background="Ablation study: profile not generated.",
                core_specialties="Ablation study: profile not generated."
            ) for i in range(n)
        ]
        research_focuses = ["Ablation study" for _ in range(n)]
        output = TriageOutput(
            difficulty=forced_level,
            justification="Ablation: triage disabled, forced level.",
            specialties=specialties,
            weights=weights,
            job_titles=job_titles,
            expert_profiles=expert_profiles,
            research_focuses=research_focuses
        )
        return TriageRunResult(response=output, usage=Usage())
    
    forced_level = getattr(cfg.triage, 'forced_level', None)
    
    def create_triage_agent(difficulty_level: str):
        config = cfg.triage[difficulty_level]
        n = config['num_experts']
        fields = get_medical_specialties(cfg)      # config: orchestrate.medical_specialties
        
        return Agent(
            name=f"TriageAgent_{difficulty_level.capitalize()}",
            model=cfg.execution.model.name,                  # config: model.name
            instructions=(
                f"{_default_cfg.triage_system_prompt}\n"
                f"{_default_cfg.triage_template.format(question=question, fields=fields, n=n)}"
            ),
            model_settings=ModelSettings(
                temperature=cfg.execution.model.temperature, # config: model.temperature
            ),
            output_type=TriageOutput,
        )

    triage_agents = {
        'easy': create_triage_agent('easy'),
        'medium': create_triage_agent('medium'),
        'hard': create_triage_agent('hard')
    }
    
    if forced_level and forced_level in triage_agents:
        result = await Runner.run(
            starting_agent=triage_agents[forced_level],
            input=question,
            context={},
        )
    else:
        triage_agent = Agent(
            name="TriageAgent",
            model=cfg.execution.model.name,
            instructions=(
                f"{_default_cfg.difficulty_system_prompt}\n"
                f"{_default_cfg.difficulty_template.format(question=question)}"
            ),
            model_settings=ModelSettings(
                temperature=cfg.execution.model.temperature,
            ),
            handoffs=[
                handoff(
                    agent=triage_agents['easy'],
                    tool_name_override="transfer_to_triage_agent_easy",
                    tool_description_override="Transfer to the triage agent for easy questions after assessing difficulty as 'easy'."
                ),
                handoff(
                    agent=triage_agents['medium'],
                    tool_name_override="transfer_to_triage_agent_medium",
                    tool_description_override="Transfer to the triage agent for medium questions after assessing difficulty as 'medium'."
                ),
                handoff(
                    agent=triage_agents['hard'],
                    tool_name_override="transfer_to_triage_agent_hard",
                    tool_description_override="Transfer to the triage agent for hard questions after assessing difficulty as 'hard'."
                )
            ]
        )
        
        result = await Runner.run(
            starting_agent=triage_agent,
            input=question,
            context={},
        )

    total_usage = Usage()
    for raw_response in result.raw_responses:
        total_usage.add(raw_response.usage)
    return TriageRunResult(
        response=result.final_output,
        usage=total_usage
    )
