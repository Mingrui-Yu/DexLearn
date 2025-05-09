# from DiffDock https://github.com/gcorso/DiffDock
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIN_EPS, MAX_EPS, N_EPS = 1e-3, 2.5, 3000
X_N = 5000
EPS_TO_SCALE = np.sqrt(2)

"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:]


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    if eps < 1:
        omega = omega.astype(np.float128)
        eps = eps.astype(np.float128) ** 2
        part1 = np.sqrt(np.pi) * (eps ** -1.5)
        part2_1 = ((omega - 2*np.pi)*(np.e ** (np.pi*omega/eps-(np.pi**2) / eps+eps/4 - ((omega/2) ** 2)/eps)) +
                   (omega + 2*np.pi)*(np.e ** (-np.pi*omega/eps-(np.pi**2) / eps+eps/4 - ((omega/2) ** 2)/eps)))
        part2_2 = (omega * (np.e ** (eps/4 - ((omega/2) ** 2)/eps)) -
                   part2_1) / (2 * np.sin(omega/2))
        p = part1 * part2_2
        p = p.astype(np.float64)
    else:
        for l in range(L):
            p += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * \
                np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


# if marginal, density over [0, pi], else over SO(3)
def _density(expansion, omega, marginal=True):
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / np.pi ** 2


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    if eps < 1:
        omega = omega.astype(np.float128)
        eps = eps.astype(np.float128) ** 2
        part2_1 = ((omega - 2*np.pi)*(np.e ** (np.pi*omega/eps-(np.pi**2) / eps)) +
                   (omega + 2*np.pi)*(np.e ** (-np.pi*omega/eps-(np.pi**2) / eps)))
        part2_2 = (omega * (np.e ** (0)) -
                   part2_1) / (2 * np.sin(omega/2))
        part1 = 1/part2_2
        dpart2_1 = (np.e ** (np.pi*omega/eps-(np.pi**2) / eps)) + \
            (omega - 2*np.pi)*(np.e ** (np.pi*omega/eps-(np.pi**2) / eps))*(np.pi/eps - omega/2/eps) + \
            (np.e ** (-np.pi*omega/eps-(np.pi**2) / eps)) + \
            (omega + 2*np.pi)*(np.e ** (-np.pi*omega/eps-(np.pi**2) /
                                        eps))*(-np.pi/eps-omega/2/eps)
        part2_left = omega * (np.e ** (0))
        dpart2_left = (np.e ** (0)) + \
            omega * (np.e ** (0)) * (-omega/2/eps)
        result = part1 * ((dpart2_left-dpart2_1) *
                          (2*np.sin(omega/2)) - (part2_left-part2_1) * (np.cos(omega/2))) / ((2*np.sin(omega/2))**2)
        result = result.astype(np.float64)
        return result
    else:
        for l in range(L):
            hi = np.sin(omega * (l + 1 / 2))
            dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
            lo = np.sin(omega / 2)
            dlo = 1 / 2 * np.cos(omega / 2)
            dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps**2) * \
                (lo * dhi - hi * dlo) / lo ** 2
        return dSigma / exp


def get_precomputed():

    if os.path.exists('.so3_omegas_array2.npy'):
        _omegas_array = np.load('.so3_omegas_array2.npy')
        _cdf_vals = np.load('.so3_cdf_vals2.npy')
        _score_norms = np.load('.so3_score_norms2.npy')
        _exp_score_norms = np.load('.so3_exp_score_norms2.npy')
        _scale_means = np.load('.so3_scale_means.npy')
    else:
        print("Precomputing and saving to cache SO(3) distribution table")
        _eps_array = 10 ** np.linspace(np.log10(MIN_EPS),
                                       np.log10(MAX_EPS), N_EPS)
        _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

        _exp_vals = np.asarray([_expansion(_omegas_array, eps)
                                for eps in _eps_array])
        _pdf_vals = np.asarray(
            [_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals])
        _scale_means = (_pdf_vals * _omegas_array).sum(-1) / X_N * np.pi
        _cdf_vals = np.asarray(
            [_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
        _score_norms = np.asarray(
            [_score(_exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))])

        _exp_score_norms = np.sqrt(
            np.sum(_score_norms**2 * _pdf_vals, axis=1) / np.sum(_pdf_vals, axis=1) / np.pi)

        np.save('.so3_omegas_array2.npy', _omegas_array)
        np.save('.so3_cdf_vals2.npy', _cdf_vals)
        np.save('.so3_score_norms2.npy', _score_norms)
        np.save('.so3_exp_score_norms2.npy', _exp_score_norms)
        np.save('.so3_scale_means.npy', _scale_means)

    _omegas_array_torch = torch.from_numpy(_omegas_array)
    _cdf_vals_torch = torch.from_numpy(_cdf_vals)
    _score_norms_torch = torch.from_numpy(_score_norms)
    _exp_score_norms_torch = torch.from_numpy(_exp_score_norms)
    _scale_means_torch = torch.from_numpy(_scale_means)

    return _omegas_array_torch, _cdf_vals_torch, _score_norms_torch, _exp_score_norms_torch, _scale_means_torch


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """ from  https://github.com/pytorch/pytorch/issues/50334#issuecomment-1247611276

    One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])  # slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    # torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.linspace(
        0, indicies.shape[0]-1, indicies.shape[0], device=indicies.device).to(torch.long)
    line_idx = line_idx.unsqueeze(-1).repeat(1, indicies.shape[-1])
    # idx = torch.cat([line_idx, indicies] , 0)
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]
