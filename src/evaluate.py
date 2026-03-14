import os
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from globals import ensure_repo_root

ensure_repo_root()

from src.models.VAE import VAE
from src.utils.data_loader import get_dataloaders
from src.utils.metrics import compute_fid_kid
from src.utils.seed_setter import set_global_seed


SamplerFn = Callable[[int, torch.device], torch.Tensor]


@dataclass
class EvalConfig:
    seed: int = 11
    batch_size: int = 64
    num_workers: int = 2
    kaggle_root: str = "data"
    use_subset: bool = False
    subset_mode: str = "csv"
    subset_csv_path: str = "provided/student_start_pack/training_20_percent.csv"
    subset_seed: int = 11
    num_samples: int = 5000
    metrics_batch_size: int = 32
    checkpoint_path: str = ""
    latent_dim: int = 128
    base_channels: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def sample_real_images(config: EvalConfig) -> np.ndarray:
    # This function samples real images from the training set of ArtBench-10 according to the provided configuration.
    train_loader, _, _ = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_subset=config.use_subset,
        subset_mode=config.subset_mode,
        subset_csv_path=config.subset_csv_path,
        subset_seed=config.subset_seed,
        kaggle_root=config.kaggle_root,
        shuffle_train=True,
    )

    # Iterate through training data loader and collect images until we have enough samples.
    images = []
    total = 0
    with tqdm(total=config.num_samples, desc="real images", unit="img") as pbar:
        for batch, _ in train_loader:
            batch = batch.cpu().numpy()
            images.append(batch)
            total += batch.shape[0]
            pbar.update(batch.shape[0])
            if total >= config.num_samples:
                break

    # Concatenate collected batches and trim to the desired number of samples.
    images = np.concatenate(images, axis=0)[: config.num_samples]
    return images


def sample_fake_images(config: EvalConfig, sampler_fn: SamplerFn) -> np.ndarray:
    device = torch.device(config.device)
    samples = []

    # Sample fake images using the provided sampler function until we have enough samples.
    remaining = config.num_samples
    with tqdm(total=config.num_samples, desc="fake images", unit="img") as pbar:
        while remaining > 0:
            cur = min(config.batch_size, remaining)

            # Sample a batch of fake images using the sampler function. The sampler function takes the number of samples to generate and the device to use.
            with torch.no_grad():
                batch = sampler_fn(cur, device)

            samples.append(batch.cpu().numpy())
            remaining -= cur
            pbar.update(cur)
    return np.concatenate(samples, axis=0)


def _find_latest_checkpoint() -> str:
    outputs_dir = "outputs"
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError("outputs/ directory not found")

    candidates = []
    for name in os.listdir(outputs_dir):
        if not name.startswith("run_vae_"):
            continue
        ckpt_path = os.path.join(outputs_dir, name, "vae.pt")
        if os.path.isfile(ckpt_path):
            candidates.append(ckpt_path)

    if not candidates:
        raise FileNotFoundError("No VAE checkpoints found under outputs/run_vae_*")

    return max(candidates, key=os.path.getmtime)


def _load_vae(checkpoint_path: str, latent_dim: int, base_channels: int, device: torch.device) -> VAE:
    model = VAE(latent_dim=latent_dim, base_channels=base_channels).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def evaluate(config: EvalConfig, sampler_fn: SamplerFn) -> Tuple[float, float, float]:
    set_global_seed(config.seed)

    # get real and fake images
    real_images = sample_real_images(config)
    fake_images = sample_fake_images(config, sampler_fn)

    # compute metrics
    fid, kid_mean, kid_std = compute_fid_kid(
        real_images,
        fake_images,
        device=config.device,
        batch_size=config.metrics_batch_size,
    )
    return fid, kid_mean, kid_std


def main():
    config = EvalConfig()
    ckpt_path = config.checkpoint_path or _find_latest_checkpoint()
    device = torch.device(config.device)
    model = _load_vae(ckpt_path, config.latent_dim, config.base_channels, device)

    def vae_sampler(num_samples: int, device: torch.device) -> torch.Tensor:
        return model.sample(num_samples, device)

    fid, kid_mean, kid_std = evaluate(config, vae_sampler)
    print({"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std})


if __name__ == "__main__":
    main()
