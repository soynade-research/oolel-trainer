# Oolel-trainer

This repository contains the training pipeline for Oolel-small, a family of Wolof language models designed for on-device usage and deployment.

Part of the Oolel family, this model is designed for high efficiency on resource-constrained devices. It is trained on a mix of open-source data and high-quality synthetic data distilled from our larger Oolel-7B models using our open-source [Oolel-translator](https://github.com/soynade-research/oolel-translator).

## Requirements

- Python 3.11+
- CUDA-capable GPU
- `uv` package manager

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv sync && uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Training

We recommend running inside a tmux session:

```bash
apt-get update && apt-get install -y tmux
tmux new -s oolel-training
```

Then launch:

```bash
sh train.sh
```

All hyperparameters are configured directly in `train.sh`.

## Monitoring

```bash
watch -n 1 nvidia-smi
tensorboard --logdir output/oolel-small
```

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@misc{oolel-small,
  author = {Soynade Research},
  title  = {Oolel-small: Small and frugal Wolof Language Models},
  year   = {2026},
  url    = {https://github.com/soynade-research/oolel-trainer}
}
```
