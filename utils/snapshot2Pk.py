import torch
import torch.fft as fft
import math 
class ParticleSpectrum:
    def __init__(self, box_size, grid_size):
        self.box_size  = float(box_size)   # L [Mpc/h]
        self.grid_size = int(grid_size)    # N (per side)

    @torch.no_grad()
    def compute_density(self, pos, par_id=None, device='cuda'):

        n = self.grid_size
        L = self.box_size
        dx = L / n

        pos = (pos.to(device) % L).contiguous().to(torch.float32)
        if par_id is not None:
            par_id = par_id.to(device)
            sort = torch.argsort(par_id)
            pos = pos[sort]

        grid = torch.zeros((n, n, n), device=device, dtype=torch.float32)

        q = pos / dx
        i0 = torch.floor(q).to(torch.long)
        f  = (q - i0).to(torch.float32)

        x0, y0, z0 = i0[:, 0] % n, i0[:, 1] % n, i0[:, 2] % n
        x1, y1, z1 = (x0 + 1) % n, (y0 + 1) % n, (z0 + 1) % n

        wx0, wy0, wz0 = 1.0 - f[:, 0], 1.0 - f[:, 1], 1.0 - f[:, 2]
        wx1, wy1, wz1 = f[:, 0],       f[:, 1],       f[:, 2]

        contribs = [
            (x0, y0, z0, wx0 * wy0 * wz0),
            (x1, y0, z0, wx1 * wy0 * wz0),
            (x0, y1, z0, wx0 * wy1 * wz0),
            (x0, y0, z1, wx0 * wy0 * wz1),
            (x1, y1, z0, wx1 * wy1 * wz0),
            (x1, y0, z1, wx1 * wy0 * wz1),
            (x0, y1, z1, wx0 * wy1 * wz1),
            (x1, y1, z1, wx1 * wy1 * wz1),
        ]
        for xx, yy, zz, ww in contribs:
            grid.index_put_((xx, yy, zz), ww, accumulate=True)

        delta = grid / grid.mean() - 1.0
        delta = delta - delta.mean()
        return delta

    @torch.no_grad()
    def power_spectrum(self, density_field):

        n = self.grid_size
        L = self.box_size

        X = fft.fftn(density_field)
        P3D = (L ** 3) * (X.real**2 + X.imag**2) / (n ** 6)
        return P3D

    @torch.no_grad()
    def compute_power_spectrum(self, pos, par_id=None, device='cuda', nbins=40):

        delta = self.compute_density(pos, par_id=par_id, device=device)
        P3D   = self.power_spectrum(delta)                     # (N,N,N), 非负

        n = int(self.grid_size)
        L = float(self.box_size)
        d = L / n

        freq = torch.fft.fftfreq(n, d=d, device=device)        # cycles / (Mpc/h)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing='ij')
        k_mag = 2.0 * torch.pi * torch.sqrt(kx**2 + ky**2 + kz**2)

        k_flat = k_mag.reshape(-1)
        P_flat = P3D.reshape(-1)

        k_f  = 2.0 * math.pi / L
        k_ny = math.pi * n / L
        k_f_t  = torch.tensor(k_f,  device=device, dtype=k_flat.dtype)
        k_ny_t = torch.tensor(k_ny, device=device, dtype=k_flat.dtype)

        m = (k_flat >= k_f_t) & torch.isfinite(P_flat)
        k_flat = k_flat[m]
        P_flat = P_flat[m]

        k_bins = torch.logspace(torch.log10(k_f_t), torch.log10(k_ny_t),
                                steps=nbins + 1, device=device, dtype=k_flat.dtype)

        k_list, p_list = [], []
        for i in range(nbins):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
            if mask.any():
                kc = torch.sqrt(k_bins[i] * k_bins[i + 1])
                k_list.append(kc)
                p_list.append(P_flat[mask].mean())

        if not k_list:
            raise RuntimeError("No valid k-modes found in any bins.")

        k_vals = torch.stack(k_list)
        ps_avg = torch.stack(p_list)

        return k_vals, ps_avg


    @torch.no_grad()
    def compute_power_spectrum_from_delta(self, delta,  device='cuda', nbins=40):

        P3D   = self.power_spectrum(delta)

        n = int(self.grid_size)
        L = float(self.box_size)
        d = L / n

        freq = torch.fft.fftfreq(n, d=d, device=device)        # cycles / (Mpc/h)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing='ij')
        k_mag = 2.0 * torch.pi * torch.sqrt(kx**2 + ky**2 + kz**2)

        k_flat = k_mag.reshape(-1)
        P_flat = P3D.reshape(-1)

        k_f  = 2.0 * math.pi / L
        k_ny = math.pi * n / L
        k_f_t  = torch.tensor(k_f,  device=device, dtype=k_flat.dtype)
        k_ny_t = torch.tensor(k_ny, device=device, dtype=k_flat.dtype)

        m = (k_flat >= k_f_t) & torch.isfinite(P_flat)
        k_flat = k_flat[m]
        P_flat = P_flat[m]

        k_bins = torch.logspace(torch.log10(k_f_t), torch.log10(k_ny_t),
                                steps=nbins + 1, device=device, dtype=k_flat.dtype)

        k_list, p_list = [], []
        for i in range(nbins):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
            if mask.any():
                kc = torch.sqrt(k_bins[i] * k_bins[i + 1])
                k_list.append(kc)
                p_list.append(P_flat[mask].mean())

        if not k_list:
            raise RuntimeError("No valid k-modes found in any bins.")

        k_vals = torch.stack(k_list)
        ps_avg = torch.stack(p_list)

        return k_vals, ps_avg


    @torch.no_grad()
    def compute_power_spectrum_from_rho(self, rho,  device='cuda', nbins=40):

        rho_bar = rho.mean()
        delta = rho / rho_bar - 1.0

        P3D   = self.power_spectrum(delta)

        n = int(self.grid_size)
        L = float(self.box_size)
        d = L / n

        freq = torch.fft.fftfreq(n, d=d, device=device)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing='ij')
        k_mag = 2.0 * torch.pi * torch.sqrt(kx**2 + ky**2 + kz**2)

        k_flat = k_mag.reshape(-1)
        P_flat = P3D.reshape(-1)

        k_f  = 2.0 * math.pi / L
        k_ny = math.pi * n / L
        k_f_t  = torch.tensor(k_f,  device=device, dtype=k_flat.dtype)
        k_ny_t = torch.tensor(k_ny, device=device, dtype=k_flat.dtype)

        # 仅保留 k ≥ k_f 的模，去掉非数值
        m = (k_flat >= k_f_t) & torch.isfinite(P_flat)
        k_flat = k_flat[m]
        P_flat = P_flat[m]

        # 4) 对数分箱，并用几何平均作为箱心
        k_bins = torch.logspace(torch.log10(k_f_t), torch.log10(k_ny_t),
                                steps=nbins + 1, device=device, dtype=k_flat.dtype)

        k_list, p_list = [], []
        for i in range(nbins):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
            if mask.any():  # 只收非空 bin，避免前端一串 0
                kc = torch.sqrt(k_bins[i] * k_bins[i + 1])     # 几何中心
                k_list.append(kc)
                p_list.append(P_flat[mask].mean())

        if not k_list:  # 所有 bin 都为空的极端情况
            raise RuntimeError("No valid k-modes found in any bins.")

        k_vals = torch.stack(k_list)
        ps_avg = torch.stack(p_list)

        return k_vals, ps_avg