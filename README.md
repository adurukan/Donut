# Donut
Donut using DVC and Flask

## Create Virtual Environment and install libraries
- python3.10 -m venv ../.venvs/donut
- source ../.venvs/donut/bin/activate
- pip install transformers # for CPU support only, pip install transformers[torch]
- pip install sentencepiece
- pip install protobuf==3.20.1