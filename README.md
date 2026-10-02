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
See [output format](docs/output.md). All runtime configuration is in
[`conf/config.yaml`](conf/config.yaml); the existing dotted override keys remain
unchanged.

## Experiment suites

One script covers `medagents`, `medrag`, `hard`, `triage`, `orchestrate`, and
`search`. It uses Hydra's multirun support and stops on the first failed suite
command. Preview a run before submitting a large sweep:

```bash
DRY_RUN=1 MODEL=gpt-4o DATASETS=medqa RUN_IDS=0 scripts/experiments.sh search
MODEL=gpt-4o DATASETS=medqa RUN_IDS=0 scripts/experiments.sh medrag
scripts/experiments.sh --help
```

`DATASETS` and `RUN_IDS` accept comma-separated values. `MODEL`, `SPLIT`, and
`PYTHON` override defaults; additional arguments are passed through as Hydra
overrides. Without overrides, the historical dataset/run/model choices are
retained. Forced agent-count ablations now bypass triage so their settings take
effect. Check that the selected model is available on your endpoint.

## Repository layout

- Root Python modules: CLI, experiment runner, agents, and retrieval runtime.
- `conf/config.yaml`: triage, discussion, search, and execution settings.
- `scripts/`: experiment suites and dataset preparation.
- `retrieval/`: Milvus startup/configuration and corpus import.
- `analysis/`: result, expert-profile, and vote-entropy analysis.
- `figures/`: one module per composite figure, shared style, and source diagrams.
- `data/`: benchmark JSONL splits; `notebooks/`: exploratory research.
- `docs/`: output format, figure instructions, and paper draft.
- `output/`: generated experiments and figures (ignored by Git).

Run research commands from the repository root, for example
`python analysis/results.py --help` or `python -m figures.main_comparison`.
See [figure inputs and commands](docs/figures.md). Removed prototypes and generated
example PDFs remain in Git history; the latest server figure sources and data
were also archived separately during recovery.

## Check

```bash
python test_smoke.py
```

This offline check uses simulated model responses and a simulated Milvus client.
It covers the CLI, discussion modes, retrieval settings, error propagation, and
experiment save/resume. Real model and Milvus access must be checked separately.
