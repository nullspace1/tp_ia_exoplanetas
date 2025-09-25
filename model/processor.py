import torch
import torch.nn as nn

class FiLMVector(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * feat_dim)
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (batch, feat_dim)         latent/features
        cond: (batch, cond_dim)         conditioning params
        """
        gb = self.mlp(cond)
        gamma, beta = gb.chunk(2, dim=1)
        return gamma * x + beta


class LightCurveProcessorFiLM(nn.Module):
    def __init__(self, latent_dim: int, params_dim: int = 5, post_proj: int | None = None):
        super().__init__()
        self.film = FiLMVector(cond_dim=params_dim, feat_dim=latent_dim)
        # Optional projection after modulation
        self.post = None
        if post_proj is not None and post_proj != latent_dim:
            self.post = nn.Sequential(
                nn.ReLU(),
                nn.Linear(latent_dim, post_proj)
            )

    def forward(self, latent_vector, stellar_temp, stellar_radius, right_ascension, declination, apparent_magnitude):
        params = torch.cat([
            stellar_temp.view(-1, 1),
            stellar_radius.view(-1, 1),
            right_ascension.view(-1, 1),
            declination.view(-1, 1),
            apparent_magnitude.view(-1, 1)
        ], dim=1)
        x = self.film(latent_vector, params)
        if self.post is not None:
            x = self.post(x)
        return x


if __name__ == "__main__":
    B = 32
    latent_dim = 256
    proc = LightCurveProcessorFiLM(latent_dim=latent_dim, params_dim=5, post_proj=None)

    z = torch.randn(B, latent_dim)
    teff = torch.randn(B)
    srad = torch.randn(B)
    ra = torch.randn(B)
    dec = torch.randn(B)
    kepmag = torch.randn(B)

    z_mod = proc(z, teff, srad, ra, dec, kepmag)  # (B, latent_dim)
    print(z_mod.shape)
