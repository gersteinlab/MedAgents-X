# Development

Read README.md for setup and entry points. Run `python test_smoke.py` after runtime
changes. The check is offline; live model and retrieval checks require `.env` and
an existing Milvus corpus. Never commit credentials, corpus files, or database
volumes. Keep experiment output formats compatible with OUTPUT_STRUCTURE.md.

Runtime flow: main.py / run_experiments.py → medagents.py → triage.py, expert.py,
orchestrate.py. Search lives in search_tools.py and retriever.py. Hydra config
lives in conf/. The experiment runner stores results through experiment.py.
