import torch
from torch import nn
from torch.nn import functional as F

# Simple VAE if Beta is 1, otherwise it's a Beta-VAE which encourages disentanglement in the latent space at the cost of reconstruction quality.

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 256, img_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # (32x32) -> (16x16) -> (8x8) -> (4x4) -> (2x2)
        self.enc_out_dim = base_channels * 8 * 2 * 2

        # This layer maps the encoder output to the latent dimension
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)

        # This layer maps the encoder output to the log variance of the latent distribution
        # We use logvar instead of std for numerical stability (exponentiate to get valid variance and avoid negative values, also simplifies the KL term!)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def encode(self, x: torch.Tensor):
        # Pass through encoder CNN and flatten
        h = self.encoder(x).view(x.size(0), -1)

        # Get latent mean
        mu = self.fc_mu(h)

        # Get latent log variance
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick: sample from N(0, 1), scale by std and shift by mean

        # compute standard deviation from log variance
        std = torch.exp(0.5 * logvar)

        # sample epsilon from standard normal
        eps = torch.randn_like(std)

        # return the reparameterized latent vector (scale and shift)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Decode from latent space back to image space
        h = self.fc_dec(z).view(z.size(0), self.base_channels * 8, 2, 2)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        # Full forward pass: encode, reparameterize, decode

        # Encode
        mu, logvar = self.encode(x)

        # Reparameterize to get latent vector
        z = self.reparameterize(mu, logvar)

        # Decode to get reconstruction
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # Sample from the latent space and decode to get new images
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):

    # Reconstruction loss (MSE between input and reconstruction)
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    # Kullback-Leibler divergence between the latent distribution and the standard normal prior
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Get total loss, weighting the KL term by beta (beta-VAE)
    # If beta = 1, it's the standard VAE loss. If beta > 1, it encourages more disentanglement at the cost of reconstruction quality.
    total = recon_loss + beta * kl_loss
    metrics = {
        "recon_loss": float(recon_loss.item()),
        "kl_loss": float(kl_loss.item()),
    }
    return total, metrics
