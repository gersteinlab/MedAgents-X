import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from expert import ExpertResult
from agents import Agent, Runner, ModelSettings, RunContextWrapper, Usage

logger = logging.getLogger(__name__)

class OrchestratorFeedback(BaseModel):
    """Feedback from orchestrator to experts for next round."""
    expert_name: str = Field(..., description="Name of the expert receiving feedback")
    feedback: str = Field(..., description="Specific feedback and suggestions for the expert")
    areas_to_explore: List[str] = Field(default_factory=list, description="Specific areas the expert should explore further")
    concerns: List[str] = Field(default_factory=list, description="Concerns about the expert's reasoning")

class OrchestratorResponse(BaseModel):
    """Response from orchestrator agent analyzing expert responses."""
    round_summary: str = Field(..., description="Overall summary of the round")
    expert_feedback: List[OrchestratorFeedback] = Field(..., description="Individual feedback for each expert")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from the round")
    areas_of_agreement: List[str] = Field(default_factory=list, description="Areas where experts agree")
    areas_of_disagreement: List[str] = Field(default_factory=list, description="Areas where experts disagree")
    should_continue: bool = Field(..., description="Whether the discussion should continue to another round")
    confidence_in_decision: str = Field(..., description="Confidence level in the current decision (low/medium/high)")

@dataclass
class OrchestratorRunResult:
    """Result from orchestrator agent run."""
    response: OrchestratorResponse
    usage: Usage

@dataclass
class OrchestratorContext:
    """Context object for orchestrator agent runs."""
    expert_results: List[Any]  # List of ExpertRes  ult from expert module
    question: str
    options: Dict[str, str]
    round_num: int
    current_decision: Dict[str, Any]

def format_question(question: str, options: Dict[str, str]) -> str:
    """Format question and options for all agents (standardized)."""
    text = f"{question}\n\n"
    for key, value in options.items():
        text += f"({key}) {value}\n"
    return text

def _create_orchestrator_agent(cfg: DictConfig) -> Agent[OrchestratorContext]:
    """Create the orchestrator agent for analyzing expert responses."""
    def get_orchestrator_instructions(context_wrapper: RunContextWrapper[OrchestratorContext], _: Agent[OrchestratorContext]) -> str:
        return """You are an expert medical orchestrator who analyzes responses from medical specialists and provides constructive feedback for multi-round discussions.

Your role is to:
1. Analyze each expert's justification and reasoning
2. Identify strengths and weaknesses in their arguments
3. Provide specific, actionable feedback for improvement
4. Highlight areas of agreement and disagreement
5. Suggest areas for further exploration
6. Evaluate the current decision based on expert responses
7. Decide whether the discussion should continue or if the current consensus is sufficient

When deciding whether to continue:
- Continue if there are significant disagreements that need resolution
- Continue if the confidence in the current decision is low
- Continue if important areas haven't been explored
- Stop if there's strong consensus and high confidence
- Stop if additional rounds are unlikely to improve the decision

Be constructive, specific, and focus on improving the quality of medical reasoning."""

    return Agent[OrchestratorContext](
        name="OrchestratorAgent",
        model=cfg.orchestrate.orchestrator.model,  # config: orchestrate.orchestrator_model or model.name
        instructions=get_orchestrator_instructions,
        output_type=OrchestratorResponse,
        model_settings=ModelSettings(temperature=cfg.orchestrate.orchestrator.temperature)  # config: orchestrate.orchestrator_temperature or model.temperature
    )

async def run_orchestrator_agent(expert_results: List[ExpertResult], question: str, options: Dict[str, str], round_num: int, current_decision: Dict[str, Any], cfg: DictConfig) -> OrchestratorRunResult:
    """Run orchestrator agent to analyze expert responses and provide feedback."""
    expert_summaries = [
        f"""
Expert: {er.profile.get('name', 'Unknown')} ({er.profile.get('job_title', 'Unknown')})
Answer: {er.result.response.answer}
Confidence: {er.result.response.confidence}
Justification: {er.result.response.justification}
Weight: {er.weight}
"""
        for er in expert_results
    ]

    analysis_prompt = f"""
Question: {question}

Options:
{chr(10).join(f'({k}) {v}' for k, v in options.items())}

Round {round_num + 1} Expert Responses:
{''.join(expert_summaries)}

Current Decision Based on Expert Responses:
Final Answer: {current_decision['final_answer']}
Vote Scores: {current_decision['vote_scores']}
Expert Details: {current_decision['details']}

Please analyze these expert responses and the current decision, then provide:
1. A summary of this round
2. Individual feedback for each expert
3. Key insights from the round
4. Areas where experts agree
5. Areas where experts disagree
6. Whether the discussion should continue to another round
7. Your confidence level in the current decision

Focus on the quality of medical reasoning, evidence cited, and whether additional discussion would improve the decision quality.
"""

    context = OrchestratorContext(
        expert_results=expert_results,
        question=question,
        options=options,
        round_num=round_num,
        current_decision=current_decision
    )

    orchestrator_agent = _create_orchestrator_agent(cfg)
    result = await Runner.run(
        starting_agent=orchestrator_agent,
        input=analysis_prompt,
        context=context,
        max_turns=cfg.orchestrate.orchestrator.max_turns
    )

    if not isinstance(result.final_output, OrchestratorResponse):
        raise TypeError("Orchestrator did not return OrchestratorResponse")
    total_usage = Usage()
    for raw_response in result.raw_responses:
        total_usage.add(raw_response.usage)
    return OrchestratorRunResult(result.final_output, total_usage)
