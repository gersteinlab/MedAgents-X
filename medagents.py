import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import uuid

from triage import run_triage_agent, TriageOutput
from expert import run_expert_agent, ExpertRunResult, ExpertResult, create_search_tool_instance, _make_search_tool_config
from orchestrate import (
    OrchestratorResponse, format_question, run_orchestrator_agent
)
from agents import Usage

logger = logging.getLogger(__name__)

@dataclass
class DebateRound:
    round_num: int
    expert_results: List[ExpertResult]
    current_decision: Dict[str, Any]
    orchestrator_feedback: Optional[OrchestratorResponse] = None

@dataclass
class AgentUsage:
    """Usage statistics for a specific agent."""
    agent_name: str
    agent_type: str  # 'triage', 'expert', 'orchestrator'
    usage: Usage
    round_num: Optional[int] = None
    expert_name: Optional[str] = None

@dataclass
class MedAgentsLog:
    rounds: List[DebateRound] = field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None
    usage_stats: List[AgentUsage] = field(default_factory=list)
    total_usage: Optional[Usage] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OrchestrationLog to a nested JSON dictionary."""
        def serialize_expert_result(er):
            return {
                "profile": {
                    "name": er.profile.get('name', 'Unknown'),
                    "job_title": er.profile.get('job_title', 'Unknown'),
                    "past_experience": er.profile.get('past_experience', ''),
                    "educational_background": er.profile.get('educational_background', ''),
                    "core_specialties": er.profile.get('core_specialties', ''),
                    "research_focus": er.profile.get('research_focus', '')
                },
                "response": {
                    "thought": er.result.response.thought,
                    "answer": er.result.response.answer,
                    "confidence": er.result.response.confidence,
                    "justification": er.result.response.justification,
                    "evidences": er.result.response.evidences
                },
                "search_tool": {
                    "previous_queries": er.result.search_tool.get_previous_queries(max_length=200, max_queries=3)
                },
                "weight": er.weight,
                "round_num": er.round_num
            }

        def serialize_orchestrator_feedback(feedback):
            if not feedback:
                return None
            return {
                "round_summary": feedback.round_summary,
                "expert_feedback": [
                    {
                        "expert_name": fb.expert_name,
                        "feedback": fb.feedback,
                        "areas_to_explore": fb.areas_to_explore,
                        "concerns": fb.concerns
                    }
                    for fb in feedback.expert_feedback
                ],
                "key_insights": feedback.key_insights,
                "areas_of_agreement": feedback.areas_of_agreement,
                "areas_of_disagreement": feedback.areas_of_disagreement,
                "should_continue": feedback.should_continue,
                "confidence_in_decision": feedback.confidence_in_decision
            }

        def serialize_usage(usage_obj):
            return {
                "agent_name": usage_obj.agent_name,
                "agent_type": usage_obj.agent_type,
                "usage": {
                    "requests": usage_obj.usage.requests,
                    "input_tokens": usage_obj.usage.input_tokens,
                    "output_tokens": usage_obj.usage.output_tokens,
                    "total_tokens": usage_obj.usage.total_tokens,
                    "input_tokens_details": {
                        "cached_tokens": usage_obj.usage.input_tokens_details.cached_tokens
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": usage_obj.usage.output_tokens_details.reasoning_tokens
                    }
                },
                "round_num": usage_obj.round_num,
                "expert_name": usage_obj.expert_name
            }

        return {
            "rounds": [
                {
                    "round_num": round_obj.round_num,
                    "expert_results": [serialize_expert_result(er) for er in round_obj.expert_results],
                    "current_decision": round_obj.current_decision,
                    "orchestrator_feedback": serialize_orchestrator_feedback(round_obj.orchestrator_feedback)
                }
                for round_obj in self.rounds
            ],
            "final_decision": self.final_decision,
            "usage_stats": [serialize_usage(usage) for usage in self.usage_stats],
            "total_usage": {
                "requests": self.total_usage.requests if self.total_usage else 0,
                "input_tokens": self.total_usage.input_tokens if self.total_usage else 0,
                "output_tokens": self.total_usage.output_tokens if self.total_usage else 0,
                "total_tokens": self.total_usage.total_tokens if self.total_usage else 0,
                "input_tokens_details": {
                    "cached_tokens": self.total_usage.input_tokens_details.cached_tokens if self.total_usage else 0
                },
                "output_tokens_details": {
                    "reasoning_tokens": self.total_usage.output_tokens_details.reasoning_tokens if self.total_usage else 0
                }
            } if self.total_usage else None
        }

class MedAgents:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.log = MedAgentsLog()

    def _add_usage_stat(self, agent_name: str, agent_type: str, usage: Usage, round_num: Optional[int] = None, expert_name: Optional[str] = None):
        """Add usage statistics for an agent."""
        agent_usage = AgentUsage(
            agent_name=agent_name,
            agent_type=agent_type,
            usage=usage,
            round_num=round_num,
            expert_name=expert_name
        )
        self.log.usage_stats.append(agent_usage)

    def _calculate_total_usage(self):
        """Calculate total usage across all agents."""
        total_usage = Usage()
        for agent_usage in self.log.usage_stats:
            if agent_usage.usage is not None:
                total_usage.add(agent_usage.usage)
        self.log.total_usage = total_usage

    def _prepare_context(self, n_experts, discussion_mode):
        search_config = _make_search_tool_config(self.cfg)
        if discussion_mode == 'group_chat_with_orchestrator':
            # Shared context for group chat
            shared_session_id = str(uuid.uuid4())
            shared_search_tool = create_search_tool_instance(search_config)
            return None, None, shared_session_id, shared_search_tool
        elif discussion_mode == 'group_chat_voting_only':
            # Shared context for group chat without orchestrator
            shared_session_id = str(uuid.uuid4())
            shared_search_tool = create_search_tool_instance(search_config)
            return None, None, shared_session_id, shared_search_tool
        elif discussion_mode == 'one_on_one_sync':
            # Individual contexts for 1-on-1 sync
            session_ids = [str(uuid.uuid4()) for _ in range(n_experts)]
            search_tools = [create_search_tool_instance(search_config) for _ in range(n_experts)]
            return session_ids, search_tools, None, None
        else:
            # Default to independent mode
            return None, None, None, None

    def _build_expert_profile(self, profile, job_title, research_focus):
        """Build complete expert profile from triage components."""
        expert_profile = profile.dict() if hasattr(profile, 'dict') else dict(profile)
        expert_profile.update({'job_title': job_title, 'research_focus': research_focus})
        return expert_profile

    def _build_debate_question(self, formatted_question, discussion_mode, round_num, expert_index, prev_answers, triage, orchestrator_feedback, expert_profile):
        """Build the debate question for an expert based on discussion mode and context."""
        debate_question = formatted_question

        if discussion_mode == 'group_chat_with_orchestrator' and round_num > 0:
            # Group chat: experts can see other experts' responses
            other_answers = [
                f"{p.name}: {a.answer}" 
                for j, (p, a) in enumerate(zip(triage.expert_profiles, prev_answers)) 
                if a is not None and j != expert_index
            ]
            if other_answers:
                debate_question += "\n\nOther experts' previous answers:\n" + "\n".join(other_answers)

        elif discussion_mode == 'group_chat_voting_only' and round_num > 0:
            # Group chat without orchestrator: experts can see other experts' responses but no orchestrator feedback
            other_answers = [
                f"{p.name}: {a.answer}" 
                for j, (p, a) in enumerate(zip(triage.expert_profiles, prev_answers)) 
                if a is not None and j != expert_index
            ]
            if other_answers:
                debate_question += "\n\nOther experts' previous answers:\n" + "\n".join(other_answers)

        elif discussion_mode == 'one_on_one_sync' and round_num > 0:
            # 1-on-1 sync: experts only get orchestrator feedback, not other experts' responses
            pass  # No other experts' responses shown

        if round_num > 0 and orchestrator_feedback and discussion_mode in ['group_chat_with_orchestrator', 'one_on_one_sync']:
            expert_name = expert_profile.get('name', 'Unknown')
            expert_feedback = next((fb for fb in orchestrator_feedback.expert_feedback if fb.expert_name == expert_name), None)
            if expert_feedback:
                debate_question += f"\n\nOrchestrator feedback from previous round:\n{expert_feedback.feedback}"
                if expert_feedback.areas_to_explore:
                    debate_question += f"\nAreas to explore: {', '.join(expert_feedback.areas_to_explore)}"
                if expert_feedback.concerns:
                    debate_question += f"\nConcerns to address: {', '.join(expert_feedback.concerns)}"
        
        return debate_question

    async def _run_single_expert(self, expert_index: int, profile: Dict[str, Any], debate_question: str, 
                                session_id: Optional[str], search_tool: Optional[Any], round_num: int, difficulty_level: str) -> ExpertRunResult:
        """Run a single expert agent and return the result."""
        expert_name = profile.get('name', f'Expert_{expert_index + 1}')
        logger.info(f"Processing expert {expert_index + 1}: {expert_name} ({profile.get('job_title', 'Unknown specialty')})")
        # Pass difficulty_level to run_expert_agent so search_mode is respected
        result = await run_expert_agent(
            question=debate_question,
            expert_profile=profile,
            cfg=self.cfg,
            session_id=session_id,
            search_tool=search_tool,
            difficulty_level=difficulty_level
        )
        self._add_usage_stat(
            agent_name=expert_name,
            agent_type="expert",
            usage=result.usage,
            round_num=round_num,
            expert_name=expert_name
        )
        logger.info(f"Expert {expert_index + 1} responded: Answer={result.response.answer}, Confidence={result.response.confidence}, Evidences={result.response.evidences}")
        return result

    def _calculate_final_decision(self, expert_results: List[ExpertRunResult], options: Dict[str, str]) -> Dict[str, Any]:
        """Calculate the current decision based on expert responses."""
        vote_scores = {k: 0.0 for k in options.keys()}
        conf_map = {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        
        for er in expert_results:
            ans = er.result.response.answer
            conf = conf_map.get(er.result.response.confidence.lower(), 1.0)
            if ans not in options:
                continue
            vote_scores[ans] += er.weight * conf
        
        total = sum(vote_scores.values())
        if total > 0:
            for k in vote_scores:
                vote_scores[k] /= total
        
        final_answer = max(vote_scores.items(), key=lambda x: x[1])[0]
        return {
            'final_answer': final_answer,
            'vote_scores': vote_scores,
            'details': [
                {
                    'name': er.profile.get('name', 'Expert'),
                    'answer': er.result.response.answer,
                    'confidence': er.result.response.confidence,
                    'evidences': er.result.response.evidences,
                    'weight': er.weight
                } for er in expert_results
            ]
        }

    async def run(self, question: str, options: Dict[str, str], difficulty: str = None) -> MedAgentsLog:
        logger.info(f"Starting multi-agent discussion with {len(options)} options")
        formatted_question = format_question(question, options)
        triage_cfg = (OmegaConf.merge(self.cfg, {'triage': {'forced_level': difficulty}})
                      if difficulty else self.cfg)
        triage_result = await run_triage_agent(formatted_question, triage_cfg)
        self._add_usage_stat(
            agent_name="TriageAgent",
            agent_type="triage",
            usage=triage_result.usage
        )
        triage: TriageOutput = triage_result.response
        n_experts = len(triage.expert_profiles)
        logger.info(f"Triage selected {n_experts} experts")
        discussion_mode = getattr(self.cfg.orchestrate, 'discussion_mode', 'group_chat_with_orchestrator')
        if difficulty is None:
            difficulty = getattr(triage_result.response, 'difficulty', None) or 'medium'
        level_config = self.cfg.triage.forced_level_custom if difficulty == 'custom' else self.cfg.triage[difficulty]
        max_rounds = level_config.max_rounds
        logger.info(f"Running up to {max_rounds} rounds in {discussion_mode} mode")
        prev_answers = [None] * n_experts
        orchestrator_feedback = None
        session_ids, search_tools, shared_session_id, shared_search_tool = self._prepare_context(n_experts, discussion_mode)
        for r in range(max_rounds):
            logger.info(f"Starting round {r+1}/{max_rounds}")
            expert_tasks = []
            for i, profile in enumerate(triage.expert_profiles):
                expert_profile = self._build_expert_profile(profile, triage.job_titles[i], triage.research_focuses[i])
                if discussion_mode == 'one_on_one_sync':
                    session_id, search_tool = session_ids[i], search_tools[i]
                elif discussion_mode in ['group_chat_with_orchestrator', 'group_chat_voting_only']:
                    session_id, search_tool = shared_session_id, shared_search_tool
                else:  # independent
                    session_id, search_tool = None, None
                debate_question = self._build_debate_question(
                    formatted_question, discussion_mode, r, i, prev_answers, triage, orchestrator_feedback, expert_profile
                )
                # Pass difficulty to _run_single_expert so search_mode is respected
                expert_tasks.append(
                    self._run_single_expert(i, expert_profile, debate_question, session_id, search_tool, r, difficulty)
                )
            logger.info(f"Running {len(expert_tasks)} experts in parallel for round {r+1}")
            expert_results = await asyncio.gather(*expert_tasks)
            round_results = [
                ExpertResult(
                    profile=self._build_expert_profile(triage.expert_profiles[i], triage.job_titles[i], triage.research_focuses[i]),
                    result=result,
                    weight=triage.weights[i],
                    round_num=r
                )
                for i, result in enumerate(expert_results)
            ]
            logger.info(f"Round {r+1} completed with {len(round_results)} expert responses")
            current_decision = self._calculate_final_decision(round_results, options)
            logger.info(f"Current decision after round {r+1}: {current_decision['final_answer']}")
            
            # Only run orchestrator for modes that use it
            if discussion_mode in ['group_chat_with_orchestrator', 'one_on_one_sync']:
                logger.info(f"Orchestrator analyzing round {r+1}")
                orchestrator_result = await run_orchestrator_agent(round_results, question, options, r, current_decision, self.cfg)
                self._add_usage_stat(
                    agent_name="OrchestratorAgent",
                    agent_type="orchestrator",
                    usage=orchestrator_result.usage,
                    round_num=r
                )
                orchestrator_feedback = orchestrator_result.response
                logger.info(f"Orchestrator provided feedback for {len(orchestrator_feedback.expert_feedback)} experts")
                logger.info(f"Orchestrator decision: should_continue={orchestrator_feedback.should_continue}, confidence={orchestrator_feedback.confidence_in_decision}")
                
                # Check if orchestrator wants to continue
                should_continue = orchestrator_feedback.should_continue
            else:
                # For voting-only mode, no orchestrator feedback
                orchestrator_feedback = None
                # Continue for max_rounds or until consensus (simple implementation)
                should_continue = r < max_rounds - 1  # Continue until last round
            
            debate_round = DebateRound(round_num=r, expert_results=round_results, current_decision=current_decision, orchestrator_feedback=orchestrator_feedback)
            self.log.rounds.append(debate_round)
            prev_answers = [er.result.response for er in round_results]
            
            if not should_continue:
                logger.info(f"Discussion ending after round {r+1}")
                break
        self.log.final_decision = self.log.rounds[-1].current_decision
        self._calculate_total_usage()
        logger.info(f"Discussion completed. Final answer: {self.log.final_decision['final_answer']}")
        logger.info(f"Total usage: {self.log.total_usage.total_tokens} tokens ({self.log.total_usage.input_tokens} input, {self.log.total_usage.output_tokens} output)")
        return self.log
