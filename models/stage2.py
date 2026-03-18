import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List, Tuple

def z_to_lna(z: torch.Tensor) -> torch.Tensor:
    """z -> ln(a) where a = 1/(1+z)."""
    return -torch.log1p(z)

class GainHead(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.to_logg = nn.Linear(emb_dim, 1)
        nn.init.zeros_(self.to_logg.weight)
        nn.init.zeros_(self.to_logg.bias)

    def forward(self, emb):
        # g>0 more stable：g = exp(logg) or softplus
        logg = self.to_logg(emb)  # (B,1)
        g = torch.exp(logg).view(-1, 1, 1, 1, 1)  # broadcast to 3D field
        return g
    
class TimeEmbed(nn.Module):
    """Embed (ln a_ini, ln a_fin) -> (B, emb_dim)."""
    def __init__(self, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim),
            nn.SiLU(),
        )

    def forward(self, lna_ini: torch.Tensor, lna_fin: torch.Tensor) -> torch.Tensor:
        if lna_ini.dim() == 0: lna_ini = lna_ini[None]
        if lna_fin.dim() == 0: lna_fin = lna_fin[None]
        t = torch.stack([lna_ini, lna_fin], dim=-1)  # (B,2)
        return self.net(t)

class FiLM3D(nn.Module):
    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.to_gb = nn.Linear(emb_dim, 2 * ch)  # gamma,beta
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
    def forward(self, x, emb):
        # x: (B,C,D,H,W), emb: (B,emb_dim)
        gb = self.to_gb(emb)  # (B,2C)
        g, b = gb.chunk(2, dim=1)  # (B,C),(B,C)
        g = g[:, :, None, None, None]
        b = b[:, :, None, None, None]
        return x * (1.0 + g) + b
    

def _norm3d(num_channels: int, kind: Literal["group", "instance", "batch"] = "group", groups: int = 8):
    if kind == "group":
        g = min(groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if kind == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    return nn.BatchNorm3d(num_channels)

class SEBlock3D(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch, max(1, ch // r), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, ch // r), ch, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm="group", act="relu",
                 residual=True, use_se=False,
                 emb_dim: int = 0):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.use_se = use_se
        self.use_film = emb_dim > 0

        self.c1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = _norm3d(out_ch, norm)
        self.c2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = _norm3d(out_ch, norm)

        if act == "relu":
            self.a1 = nn.ReLU(inplace=True); self.a2 = nn.ReLU(inplace=True)
        else:
            self.a1 = nn.LeakyReLU(0.1, inplace=True); self.a2 = nn.LeakyReLU(0.1, inplace=True)

        self.film1 = FiLM3D(emb_dim, out_ch) if self.use_film else None
        self.film2 = FiLM3D(emb_dim, out_ch) if self.use_film else None

        self.se = SEBlock3D(out_ch) if use_se else nn.Identity()

    def forward(self, x, emb=None):
        out = self.c1(x); out = self.n1(out)
        if self.use_film:
            out = self.film1(out, emb)
        out = self.a1(out)

        out = self.c2(out); out = self.n2(out)
        if self.use_film:
            out = self.film2(out, emb)
        out = self.a2(out)

        out = self.se(out)
        if self.residual:
            out = out + x
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.down = nn.Conv3d(in_ch, in_ch, 3, stride=2, padding=1, bias=True)
        self.block = ConvBlock3D(in_ch, out_ch, **kw)
    def forward(self, x, emb=None):
        return self.block(self.down(x), emb)

class Up(nn.Module):
    def __init__(self, up_ch, skip_ch, out_ch, **kw):
        super().__init__()
        self.up = nn.ConvTranspose3d(up_ch, up_ch, 2, stride=2)
        self.block = ConvBlock3D(up_ch + skip_ch, out_ch, **kw)
    def forward(self, x, skip, emb=None):
        x = self.up(x)
        dx = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dz = skip.size(4) - x.size(4)
        if dx != 0 or dy != 0 or dz != 0:
            x = F.pad(x, (0, max(0,dz), 0, max(0,dy), 0, max(0,dx)))
            x = x[:, :, :skip.size(2), :skip.size(3), :skip.size(4)]
        x = torch.cat([x, skip], dim=1)
        return self.block(x, emb)


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, levels=4,
                 norm="group", act="relu", residual=True, use_se=False,
                 final_activation="none",
                 z_cond: bool = True,
                 fixed_z0: bool = True,
                 emb_dim: int = 64,
                 z0_fixed: float = 100):
        super().__init__()
        assert levels >= 2

        self.z_cond = z_cond
        self.fixed_z0 = fixed_z0
        if fixed_z0:
            self.register_buffer("_z0_fixed", torch.tensor(float(z0_fixed), dtype=torch.float32), persistent=True)
        if z_cond:
            self.z_embed = TimeEmbed(emb_dim=emb_dim, hidden=128)
        else:
            self.z_embed = None
            emb_dim = 0

        chs = [base_ch * (2 ** i) for i in range(levels)]
        self.enc0 = ConvBlock3D(in_ch, chs[0], norm=norm, act=act, residual=residual, use_se=use_se, emb_dim=emb_dim)
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i+1], norm=norm, act=act, residual=residual, use_se=use_se, emb_dim=emb_dim)
            for i in range(levels - 1)
        ])

        self.bottleneck = ConvBlock3D(chs[-1], chs[-1], norm=norm, act=act, residual=residual, use_se=use_se, emb_dim=emb_dim)

        self.ups = nn.ModuleList([
            Up(up_ch=chs[i+1], skip_ch=chs[i], out_ch=chs[i],
               norm=norm, act=act, residual=residual, use_se=use_se, emb_dim=emb_dim)
            for i in reversed(range(levels - 1))
        ])

        self.out_conv = nn.Conv3d(chs[0], out_ch, 1)
        self.out_act = nn.Identity() if final_activation == "none" else (nn.Tanh() if final_activation == "tanh" else nn.Sigmoid())

        if self.z_cond:
            self.gain_head = GainHead(emb_dim=emb_dim)   # outputs g: (B,1,1,1,1)
        else:
            self.gain_head = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, z1=None, z0=None):
        """
        fixed z0：forward(x, z1)
        various z0：forward(x, z1, z0)
        """
        emb = None
        if self.z_cond:
            if z1 is None:
                raise ValueError("z1 is required when z_cond=True")

            z1c = _to_col(z1, x).squeeze(1)  # (B,)
            if self.fixed_z0:
                z0c = _to_col(self._z0_fixed, x).squeeze(1)  # (B,)
            else:
                if z0 is None:
                    raise ValueError("z0 is required when fixed_z0=False")
                z0c = _to_col(z0, x).squeeze(1)  # (B,)

            lna0 = z_to_lna(z0c)
            lna1 = z_to_lna(z1c)
            emb = self.z_embed(lna0, lna1)  # (B, emb_dim)

        s0 = self.enc0(x, emb)
        feats = [s0]
        f = s0
        for down in self.downs:
            f = down(f, emb)
            feats.append(f)

        f = self.bottleneck(f, emb)

        for i, up in enumerate(self.ups):
            skip = feats[-(i + 2)]
            f = up(f, skip, emb)

        out = self.out_conv(f)
        out = self.out_act(out)

        if self.gain_head is not None:
            if emb is None:
                raise RuntimeError("emb is None but gain_head is set.")
            g = self.gain_head(emb)          # (B,1,1,1,1)
            out = out * g

        return out

def _to_col(z, x_like):
    B = x_like.shape[0]
    if isinstance(z, (float, int)):
        return torch.full((B, 1), float(z), device=x_like.device, dtype=x_like.dtype)
    z = z.to(device=x_like.device, dtype=x_like.dtype)
    if z.dim() == 0:
        z = z.view(1, 1).repeat(B, 1)
    elif z.dim() == 1:
        z = z.view(-1, 1)
        if z.shape[0] == 1 and B > 1:
            z = z.repeat(B, 1)
    elif z.dim() == 2 and z.shape[1] == 1:
        if z.shape[0] == 1 and B > 1:
            z = z.repeat(B, 1)
    else:
        raise ValueError(f"z has unsupported shape {tuple(z.shape)}")
    if z.shape[0] != B:
        raise ValueError(f"Batch mismatch: z has B={z.shape[0]}, x has B={B}")
    return z