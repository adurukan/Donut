# Donut
Donut using DVC and Flask

## Create Virtual Environment and install libraries
- python3.10 -m venv ../.venvs/donut
- source ../.venvs/donut/bin/activate
- pip install transformers # for CPU support only, pip install transformers[torch]
- pip install sentencepiece
- pip install protobuf==3.20.1


## Resources
- https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/donut#overview
- https://github.com/clovaai/donut
- https://huggingface.co/datasets/naver-clova-ix/cord-v2