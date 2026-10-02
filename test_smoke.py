"""Offline regression check: python test_smoke.py. No keys or Milvus required."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import hydra
import torch
from agents import Usage

import main
import run_experiments
from expert import ExpertResponse, _make_search_tool_config, create_expert_agent
from medagents import MedAgents
from orchestrate import OrchestratorResponse
from retriever import MedCPTRetriever
from search_tools import SearchTool
from triage import ExpertProfile, TriageOutput


async def fake_run(starting_agent, **kwargs):
    if starting_agent.name.startswith('TriageAgent'):
        response = TriageOutput(
            difficulty='easy', justification='Smoke test', specialties=['Medicine'],
            weights=[1.0], job_titles=['Physician'], research_focuses=['Medicine'],
            expert_profiles=[ExpertProfile(name='Test', past_experience='Test',
                educational_background='Test', core_specialties='Medicine')],
        )
    elif starting_agent.name == 'ExpertAgent':
        response = ExpertResponse(thought='Test', answer='B', confidence='high',
            evidences=['Test evidence'], justification='Test')
    else:
        response = OrchestratorResponse(round_summary='Test', expert_feedback=[],
            should_continue=False, confidence_in_decision='high')
    return SimpleNamespace(final_output=response,
        raw_responses=[SimpleNamespace(usage=Usage(requests=1, total_tokens=3))])


def check():
    with hydra.initialize_config_dir(version_base=None,
            config_dir=str(Path(__file__).resolve().parent / 'conf')):
        cfg = hydra.compose(config_name='config', overrides=[
            'execution.model.name=gpt-4o', 'search.topk.retrieve=7', 'search.topk.rerank=2',
            'search.similarity.threshold=0.75', 'search.hardware.device=cpu',
            'search.milvus.uri=http://localhost:19531',
            'search.rewrite=false', 'search.review=false',
        ])
    config = _make_search_tool_config(cfg)
    assert (config.retrieve_topk, config.rerank_topk, config.device) == (7, 2, 'cpu')
    assert config.model_name == 'gpt-4o'
    assert config.query_similarity_threshold == 0.75
    assert config.milvus_uri == 'http://localhost:19531'
    cfg.search.allowed_sources = 'all'
    assert _make_search_tool_config(cfg).allowed_sources == ['cpg', 'statpearls', 'recop', 'textbooks']
    cfg.search.allowed_sources = ['cpg', 'statpearls', 'recop', 'textbooks']
    client = Mock()
    client.list_collections.return_value = ['cpg', 'textbooks']
    client.search.return_value = [[{'entity': {'text': 'same document'}}]]
    retriever = MedCPTRetriever(client)
    retriever.encode = Mock(return_value=torch.ones(768))
    assert retriever.retrieve('test', topk=7) == ['same document']
    assert client.search.call_count == 2
    assert client.search.call_args.kwargs['limit'] == 7
    retriever.rerank = lambda query, docs: docs
    tool = SearchTool(config)
    tool._retriever = retriever
    assert asyncio.run(tool.search_medical_knowledge('test')) == ['same document']
    before = client.search.call_count
    assert asyncio.run(tool.search_medical_knowledge('test')) == ['same document']
    assert client.search.call_count == before  # similar queries reuse documents
    for mode, count in [('vector', 1), ('web', 1), ('both', 2)]:
        cfg.search.search_mode = mode
        assert len(create_expert_agent(cfg, tool, difficulty_level='hard').tools) == count
    assert not create_expert_agent(cfg, tool, difficulty_level='easy').tools

    with patch('agents.Runner.run', side_effect=fake_run), \
            patch('main.setup_openai_client'), patch('run_experiments.setup_openai_client'), \
            contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
        for mode in ['group_chat_with_orchestrator', 'group_chat_voting_only', 'one_on_one_sync']:
            cfg.orchestrate.discussion_mode = mode
            result = asyncio.run(MedAgents(cfg).run('Test?', {'A': 'No', 'B': 'Yes'}))
            assert result.final_decision['final_answer'] == 'B'
            assert result.total_usage.requests == (2 if mode.endswith('voting_only') else 3)
            json.dumps(result.to_dict())
        output = Path(tmp) / 'single.json'
        with patch('sys.argv', ['main.py', '--mode', 'single', '--question', 'Test?',
                '--options', 'A:No', 'B:Yes', '--output', str(output),
                '--config', 'search.hardware.device=cpu']):
            main.main()
        assert json.loads(output.read_text())[0]['final_answer'] == 'B'
        dataset = Path(tmp) / 'tiny'
        dataset.mkdir()
        (dataset / 'test.jsonl').write_text(json.dumps({
            'question': 'Test?', 'options': {'A': 'No', 'B': 'Yes'}, 'answer_idx': 'B'
        }) + '\n')
        cfg.execution.dataset.update(name='tiny', dir=tmp, split='test')
        cfg.execution.output.folder = str(Path(tmp) / 'output')
        run_experiments.main.__wrapped__(cfg)  # train.jsonl is intentionally absent
        summary = next((Path(tmp) / 'output').rglob('summary.json'))
        assert json.loads(summary.read_text())[0]['final_answer'] == 'B'
        run_experiments.main.__wrapped__(cfg)
        assert len(json.loads(summary.read_text())) == 1  # resume skips completed rows

    with patch('agents.Runner.run', new=AsyncMock(side_effect=RuntimeError('offline'))):
        try:
            asyncio.run(MedAgents(cfg).run('Test?', {'A': 'No', 'B': 'Yes'}))
        except RuntimeError as exc:
            assert str(exc) == 'offline'
        else:
            raise AssertionError('Model errors must propagate, not fabricate answers')
    print('PASS: CLI, three debate modes, retrieval, config, experiment save/resume, errors')


if __name__ == '__main__':
    check()
