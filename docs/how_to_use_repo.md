# How to Use This Repo

## Setup

- Ensure the ArtBench-10 Kaggle data is in `data/`:
  - `data/ArtBench-10.csv`
  - `data/artbench-10-python/artbench-10-batches-py/`

## Configs

- Training uses YAML configs in `configs/`.
- Example: `configs/vae_config.yml`
- Key fields include: `model_type`, `device`, `optimizer`, `batch_size`, `epochs`, `kaggle_root`.

## Train

```bash
python src/train.py configs/vae_config.yml
```

- Outputs are written to `outputs/run_<model>_<timestamp>/`.

## Evaluate

```bash
python src/evaluate.py
```

- Evaluation loads the latest VAE checkpoint from `outputs/run_vae_*`.
- To change sample counts, edit `src/evaluate.py` defaults.

## Add a New Model (!! Modular !! 👈🤣)

1) Implement the model in `src/models/`.
2) Add a `model_type` entry in the YAML config.
3) Extend the model selection block in `src/train.py` and raise `NotImplementedError` until implemented.
4) Add a sampler in `src/evaluate.py` for that model.
