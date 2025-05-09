import numpy as np
import torch
from torch import nn
from . import so3_utils
from pytorch3d import transforms as pttf
from .so3_utils import MIN_EPS, MAX_EPS, N_EPS, X_N, EPS_TO_SCALE


class IGSO3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        _omegas_array_torch, _cdf_vals_torch, _score_norms_torch, _exp_score_norms_torch, _scale_means_torch = so3_utils.get_precomputed()
        # (N, )
        self.register_buffer("_omegas_array", _omegas_array_torch)
        # (M, N)
        self.register_buffer("_cdf_vals", _cdf_vals_torch)
        self.register_buffer("_score_norms", _score_norms_torch)
        # (M, )
        self.register_buffer("_exp_score_norms", _exp_score_norms_torch)
        self.register_buffer("_scale_means", _scale_means_torch)

    def to_device(self, device):
        if device != self._omegas_array.device:
            self._omegas_array = self._omegas_array.to(device)
            self._cdf_vals = self._cdf_vals.to(device)
            self._score_norms = self._score_norms.to(device)
            self._exp_score_norms = self._exp_score_norms.to(device)
            self._scale_means = self._scale_means.to(device)

    def _eps_idx(self, eps):
        eps_idx = (torch.log10(eps) - np.log10(MIN_EPS)) / \
            (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
        eps_idx = torch.clip(torch.round(eps_idx).long(),
                             min=0, max=N_EPS - 1)
        return eps_idx

    def _sample(self, batch_size, eps):
        eps_idx = self._eps_idx(eps)  # (N, )
        x = torch.rand((batch_size, 1), device=eps.device, dtype=eps.dtype)
        return so3_utils.interpolate(x, self._cdf_vals[eps_idx], self._omegas_array.unsqueeze(0))

    # eps: (N,) -> samples: (N, 3)
    def _sample_vec(self, eps):
        batch_size = eps.shape[0]
        x = torch.randn((batch_size, 3), device=eps.device, dtype=eps.dtype)
        x /= x.norm(dim=-1, keepdim=True)
        return x * self._sample(batch_size, eps)

    def _score_vec(self, eps, vec):
        eps_idx = self._eps_idx(eps)
        om = torch.norm(vec, dim=-1, keepdim=True)
        return so3_utils.interpolate(om, self._omegas_array.unsqueeze(0), self._score_norms[eps_idx]) * vec / om

    def _score_norm(self, eps):
        eps_idx = self._eps_idx(eps)
        return self._exp_score_norms[eps_idx]

    def sample(self, scale):
        eps = self.scale_to_eps(scale)
        shape = eps.shape
        self.to_device(eps.device)
        return self._sample_vec(eps.reshape(-1)).reshape(*shape, 3).to(eps.dtype)

    def score_vec(self, scale, vec):
        eps = self.scale_to_eps(scale)
        shape = eps.shape
        self.to_device(eps.device)
        return self._score_vec(eps.reshape(-1), vec.reshape(-1, 3)).reshape(*shape, 3).to(eps.dtype)

    def score_norm(self, scale):
        eps = self.scale_to_eps(scale)
        shape = eps.shape
        self.to_device(eps.device)
        return self._score_norm(eps.reshape(-1)).reshape(*shape, ).to(eps.dtype)

    def scale_to_eps(self, scale):
        return scale / EPS_TO_SCALE

    def forward(self):
        raise NotImplementedError()
