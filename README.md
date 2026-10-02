# MedAgents-X

Medical multiple-choice question answering with triage, expert agents, evidence
retrieval, and optional moderated discussion.

## Setup

```bash
python -m pip install -r requirements.txt
cp .example.env .env
# Set OPENAI_API_KEY, OPENAI_ENDPOINT, and MILVUS_URI in .env.
```

The model endpoint must support the OpenAI Responses API and structured outputs.
Web search additionally requires hosted web-search support. Retrieval uses
Milvus 2.5 and the MedCPT query and cross encoders. CPU is the default; set
`DEVICE=cuda` to use a GPU. See [retrieval setup](retrieval/README.md).

## Run

```bash
python main.py                         # interactive
python main.py --mode single \
  --question 'Which organ pumps blood?' --options 'A:Heart' 'B:Liver' \
  --difficulty easy --output result.json
python main.py --mode batch --questions questions.json --output results.json
```

Batch input is a JSON list with `question`, `options`, and optional `difficulty`.
Use repeatable `--config key=value` arguments for configuration overrides:

```bash
python main.py --mode single \
  --question 'Which organ pumps blood?' --options 'A:Heart' 'B:Liver' \
  --difficulty hard --config search.search_mode=vector \
  --config search.topk.retrieve=10 --config search.topk.rerank=3
```

For dataset experiments, use Hydra overrides directly:

```bash
python run_experiments.py execution.dataset.name=medqa \
  execution.dataset.split=test_hard search.search_mode=vector
```

Only the selected JSONL split is required. Results go under
`output/<dataset>/<experiment>/run_<id>/<model>/`; rerunning resumes completed rows.
See [output format](OUTPUT_STRUCTURE.md) and the configuration in `conf/`.
The existing ablation scripts, analysis scripts, figures, notebooks, and datasets
remain available. Superseded `old/` code is retained in Git history.

## Check

```bash
python test_smoke.py
```

This offline check uses simulated model responses and a simulated Milvus client.
It covers the CLI, discussion modes, retrieval settings, error propagation, and
experiment save/resume. Real model and Milvus access must be checked separately.
