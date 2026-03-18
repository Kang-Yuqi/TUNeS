import torch
import torch.nn as nn
from typing import Literal, Tuple

def _make_center_weight_3d(
    D: int, H: int, W: int,
    inner_frac: float = 0.6,
    min_w: float = 0.2,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:

    if device is None:
        device = "cpu"

    z = torch.linspace(0, 1, D, device=device, dtype=dtype)
    y = torch.linspace(0, 1, H, device=device, dtype=dtype)
    x = torch.linspace(0, 1, W, device=device, dtype=dtype)

    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

    dist_edge = torch.minimum(
        torch.minimum(X, 1 - X),
        torch.minimum(Y, 1 - Y)
    )
    dist_edge = torch.minimum(dist_edge, Z)
    dist_edge = torch.minimum(dist_edge, 1 - Z)

    d_inner = 0.5 * inner_frac
    w = (dist_edge / d_inner).clamp(0.0, 1.0)
    w = min_w + (1.0 - min_w) * w
    return w  # (D, H, W)


def _hann_1d(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if device is None:
        device = "cpu"
    i = torch.arange(n, device=device, dtype=dtype)
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)
    return 0.5 * (1.0 - torch.cos(2.0 * torch.pi * i / (n - 1)))


def _make_hann_weight_3d(
    D: int, H: int, W: int,
    min_w: float = 0.1,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:

    if device is None:
        device = "cpu"

    wz = _hann_1d(D, device=device, dtype=dtype)
    wy = _hann_1d(H, device=device, dtype=dtype)
    wx = _hann_1d(W, device=device, dtype=dtype)

    Z, Y, X = torch.meshgrid(wz, wy, wx, indexing="ij")
    w = Z * Y * X

    w = w / w.max().clamp_min(1e-8)
    w = min_w + (1.0 - min_w) * w
    return w  # (D, H, W)


class EdgeWeightedLoss(nn.Module):


    def __init__(
        self,
        base: Literal["l1", "l2"] = "l1",
        weight_mode: Literal["center", "hann"] = "center",
        inner_frac: float = 0.6,
        min_w: float = 0.2,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 0.0 < inner_frac <= 1.0
        assert 0.0 < min_w <= 1.0

        self.base = base
        self.weight_mode = weight_mode
        self.inner_frac = inner_frac
        self.min_w = min_w
        self.eps = eps

        self.register_buffer("_weight", None, persistent=False)
        self._cached_shape: Tuple[int, int, int] | None = None

    def _build_weight(self, D: int, H: int, W: int, device, dtype):
        if self.weight_mode == "center":
            w = _make_center_weight_3d(
                D, H, W,
                inner_frac=self.inner_frac,
                min_w=self.min_w,
                device=device,
                dtype=dtype,
            )
        else:  # "hann"
            w = _make_hann_weight_3d(
                D, H, W,
                min_w=self.min_w,
                device=device,
                dtype=dtype,
            )
        self._weight = w
        self._cached_shape = (D, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, C, D, H, W)
        """
        assert pred.shape == target.shape, "pred and target should have same shape"
        assert pred.dim() == 5, "need (B, C, D, H, W)"

        B, C, D, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype

        if (self._weight is None) or (self._cached_shape != (D, H, W)):
            self._build_weight(D, H, W, device=device, dtype=dtype)
        else:
            self._weight = self._weight.to(device=device, dtype=dtype)

        w = self._weight.view(1, 1, D, H, W)

        if self.base == "l1":
            err = (pred - target).abs()
        else:
            err = (pred - target) ** 2

        loss = (err * w).sum() / (w.sum() * C + self.eps)
        return loss