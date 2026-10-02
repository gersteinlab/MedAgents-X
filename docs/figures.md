# Figures

Run these modules from the repository root. Each reads real inputs from
`figures/data/` and writes to `output/figures/`. Plotting requires the Python
requirements plus Poppler's `pdftoppm` for the source PDF diagrams.

| Command (`python -m …`) | Required inputs under `figures/data/` |
| --- | --- |
| `figures.main_comparison` | `main_comparison.csv`, `agent_configuration.csv`, `role_play.csv`, `orchestration_style.csv` |
| `figures.agent_configuration` | `agent_configuration.csv`, `orchestration_style.csv`, `vote_entropy_summary.csv`, `crosstalk.csv` |
| `figures.error_analysis` | `error_analysis/counts_by_run.json`, `error_analysis/annotator_agreement.json`, `evidence_quality/evidence_quality_scores.json` |
| `figures.dataset_comparison` | `main_comparison.csv` |
| `figures.search_ablation` | `search_ablation.csv` |
| `figures.expert_profiles` | `role_play.csv`, `expert_profiles.csv` |
| `figures.dataset_beeswarm` | `main_comparison.csv` |
| `figures.tables` | `main_comparison.csv` |

`figures/assets/` holds the two source diagrams used by the composites. The
shared palette/style lives in `style.py` and `medagents.mplstyle`. Missing input
files raise errors; empty expert profiles are not replaced with synthetic data.

The cleanup uses the latest server working-copy figures. Unmodified originals,
CSV inputs, and error/evidence analysis are in `research-figures-original.tar.gz`
in the recovery Dropbox folder. Extract that archive into a separate directory;
copy its `figures/data/` into this repository's `figures/data/`, and its
`error_analysis/` and `evidence_quality/` into `figures/data/` as well. Do not
extract the original scripts over this cleaned tree.

Analysis entrypoints are `analysis/results.py`, `analysis/experts.py`, and
`analysis/vote_entropy.py`. The preserved CSVs are historical research inputs;
rendering them does not rerun or validate the benchmark experiments.
