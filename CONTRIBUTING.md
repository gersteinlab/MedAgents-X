# Contributing

Install `requirements.txt` and follow README.md for configuration and entry
points. Make changes on a branch and run `python test_smoke.py` before opening a
pull request. Keep existing experiment output formats compatible.

Describe the behavior changed and the checks run. Distinguish offline checks
from tests against real model and Milvus services. Do not commit `.env`, database
volumes, corpus files, or generated experiment outputs.
