# Experiment output

Dataset runs write to
`output/<dataset>/<experiment>/run_<id>/<model>/`.
Hyphens in the model name become underscores (for example `gpt-4o` → `gpt_4o`).
Experiment names may contain a family/subconfiguration path, such as
`search_features/no_query_rewrite`.

Each run contains:

| File | Contents |
| --- | --- |
| `config.yaml` | Full configuration used for the run |
| `summary.json` | Ordered list of questions, answer indices, final answers, votes, difficulty, and elapsed time |
| `logs.json` | Full triage, discussion, retrieval, and decision logs, keyed by question `realidx` |
| `usage.json` | Per-question and aggregate token/request usage, plus accuracy |
| `metrics.json` | Accuracy, difficulty breakdown, usage/time aggregates, and experiment metadata |

Results are saved after each completed question. Rerunning the same dataset,
experiment, run ID, and model skips `realidx` values already in `summary.json`.
Use a different experiment name or run ID when changing a configuration, so old
rows are not treated as completed results for the new settings.

Single-question and batch CLI runs instead write their result list to the path
passed with `main.py --output`.
