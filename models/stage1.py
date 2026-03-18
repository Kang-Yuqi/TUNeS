import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def z_to_lna(z: torch.Tensor) -> torch.Tensor:
    """z -> ln(a) where a = 1/(1+z)."""
    return -torch.log1p(z)


class FourierPosEnc(nn.Module):
    """
    Fourier features for 3D position.
    x is assumed normalized to [0, 1] or [-1, 1] (either is ok; choose consistently).
    """
    def __init__(self, num_bands: int = 8, include_input: bool = True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)

        # Frequencies: 2^k
        freqs = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 3) or (N, 3)
        return: (..., C)
        """
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,N,3)

        # (B,N,3,nb)
        xb = x[..., None] * self.freqs  # broadcast
        # use 2π
        xb = 2.0 * math.pi * xb
        sin = torch.sin(xb)
        cos = torch.cos(xb)
        # (B,N,3*nb*2)
        feat = torch.cat([sin, cos], dim=-1).reshape(x.size(0), x.size(1), -1)

        if self.include_input:
            feat = torch.cat([x, feat], dim=-1)

        if orig_shape[0] != feat.shape[0] and orig_shape[0] != 1:
            # should not happen, but keep safe
            pass

        return feat

class TimeEmbed(nn.Module):
    """Embed (ln a_ini, ln a_fin) into a conditioning vector."""
    def __init__(self, dim: int = 128, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
            nn.SiLU(),
        )

    def forward(self, lna_ini: torch.Tensor, lna_fin: torch.Tensor) -> torch.Tensor:
        """
        lna_ini/lna_fin: (B,) or scalar tensor
        return: (B, dim)
        """
        if lna_ini.dim() == 0:
            lna_ini = lna_ini[None]
        if lna_fin.dim() == 0:
            lna_fin = lna_fin[None]
        t = torch.stack([lna_ini, lna_fin], dim=-1)  # (B,2)
        return self.net(t)


class FiLMLayer(nn.Module):
    """
    Feature-wise linear modulation: y = (1 + gamma) * LN(x) + beta
    gamma, beta are produced from time embedding.
    """
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta  = nn.Linear(cond_dim, feat_dim)

        # Initialize small modulation (stabilizes early training)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,C)
        cond: (B,cond_dim)
        """
        x0 = self.norm(x)
        gamma = self.to_gamma(cond).unsqueeze(1)  # (B,1,C)
        beta  = self.to_beta(cond).unsqueeze(1)   # (B,1,C)
        return (1.0 + gamma) * x0 + beta


class PerParticleCondDisplacement(nn.Module):
    """
    Per-particle conditional displacement model:
      inputs: x_ini, v_ini, z_ini, z_fin
      output: dx (displacement)
    No neighbor information. Suitable as Stage-0 / baseline.

    Notes:
    - Expect x_ini in box coordinates. You should normalize x before feeding (recommended).
    - v_ini should be normalized with global stats (recommended).
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        depth: int = 6,
        time_dim: int = 128,
        pos_fourier_bands: int = 8,
        pos_include_input: bool = True,
        use_pos_fourier: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_pos_fourier = bool(use_pos_fourier)

        if self.use_pos_fourier:
            self.pos_enc = FourierPosEnc(num_bands=pos_fourier_bands, include_input=pos_include_input)
            pos_dim = (3 if pos_include_input else 0) + 3 * pos_fourier_bands * 2
        else:
            self.pos_enc = None
            pos_dim = 3

        in_dim = pos_dim + 3

        self.time = TimeEmbed(dim=time_dim, hidden=time_dim)

        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "film": FiLMLayer(hidden_dim, time_dim),
                "ff": nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            }))

        self.out_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Optional: start near zero displacement
        nn.init.zeros_(self.out_head[-1].weight)
        nn.init.zeros_(self.out_head[-1].bias)

    def forward(
        self,
        x_ini: torch.Tensor,
        v_ini: torch.Tensor,
        z_ini: torch.Tensor,
        z_fin: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_ini: (B,N,3) or (N,3)
        v_ini: (B,N,3) or (N,3)
        z_ini/z_fin: (B,) or scalar tensor
        return:
          dx: (B,N,3) or (N,3) matching input batching
        """
        squeeze_batch = False
        if x_ini.dim() == 2:
            x_ini = x_ini.unsqueeze(0)
            v_ini = v_ini.unsqueeze(0)
            squeeze_batch = True

        B, N, _ = x_ini.shape

        lna_ini = z_to_lna(z_ini)
        lna_fin = z_to_lna(z_fin)
        cond = self.time(lna_ini, lna_fin)  # (B,time_dim)
        if cond.size(0) == 1 and B > 1:
            cond = cond.expand(B, -1)

        if self.use_pos_fourier:
            px = self.pos_enc(x_ini)  # (B,N,pos_dim)
        else:
            px = x_ini

        h = torch.cat([px, v_ini], dim=-1)          # (B,N,in_dim)
        h = self.in_proj(h)                         # (B,N,H)

        for blk in self.blocks:
            h = blk["film"](h, cond)                # (B,N,H)
            h = h + blk["ff"](h)                    # residual

        dx = self.out_head(h)                       # (B,N,3)

        return dx.squeeze(0) if squeeze_batch else dx
