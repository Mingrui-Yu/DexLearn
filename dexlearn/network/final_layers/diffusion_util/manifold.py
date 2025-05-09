import math
import numpy as np
import torch
from pytorch3d import transforms as pttf
from einops import repeat, rearrange

from .distributions.so3 import IGSO3
from .t_schedule import get_tschedule

def jacobian_matrix(f, z):
    """Calculates the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    jacobian = torch.zeros(
        (*f.shape, z.shape[-1]), device=f.device)
    for i in range(f.shape[-1]):
        jacobian[..., i, :] = torch.autograd.grad(
            f[..., i].sum(), z, retain_graph=(i != f.shape[-1]-1), allow_unused=True)[0]
    return jacobian.contiguous()


def approx_jacobian_trace(f, z):
    e = torch.normal(mean=0, std=1, size=f.shape,
                     device=f.device, dtype=f.dtype)
    grad = torch.autograd.grad(f, z, grad_outputs=e)[0]
    return torch.einsum('nka,nka->nk', e, grad)

def jacobian_trace(log_prob_type, dx, dy):
    if log_prob_type == 'accurate_cont':
        # time consuming
        jacobian_mat = jacobian_matrix(dy, dx)
        return jacobian_mat.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    elif log_prob_type == 'estimate':
        # quick
        return approx_jacobian_trace(dy, dx)
    else:
        return 0


class Manifold:
    # B: batch_size, D: euc_dim, N: rot_num
    # the manifold R^D x SO(3)^N
    # euclidean: (B, ..., D)
    # rotation score: axis angle representation (B, ..., N, 3)
    # rotation rep(resentation): matrix representation (B, ..., N, 3, 3)
    def __init__(self, config, euc_dim, rot_num) -> None:
        self.config = config
        self.euc_dim = euc_dim
        self.rot_num = rot_num
        self.so3_noise = IGSO3()
        self.t_schedule = get_tschedule(config)

    # total dimension of score
    def score_dim(self):
        return self.euc_dim + self.rot_num * 3

    # total dimension of representation of score
    def score_rep_dim(self):
        if self.config.score_rot_rep in ['aa', 'epsilon']:
            return self.euc_dim + self.rot_num * 3
        if self.config.score_rot_rep == 'est_4d':
            return self.euc_dim + self.rot_num * 4
        elif self.config.score_rot_rep in ['rel_mat', 'est_9d']:
            return self.euc_dim + self.rot_num * 9
        else:
            raise NotImplementedError()

    # pack score tensor(B, D+xN) into dict
    def score_to_dict(self, tensor, dic, scale):
        result_dict = {}
        if self.euc_dim:
            if self.config.score_euc_rep == 'score':
                raise NotImplementedError()
                result_dict['euc'] = tensor[:, :self.euc_dim]
            elif self.config.score_euc_rep == 'epsilon':
                result_dict['euc'] = -tensor[:, :self.euc_dim] / scale['euc'].unsqueeze(-1)
            elif self.config.score_euc_rep == 'est':
                raise NotImplementedError()
                est = tensor[:, :self.euc_dim]
                noise = dic['euc'] - est * (1-scale['euc'].square()).sqrt().unsqueeze(-1)
                result_dict['euc'] = -noise / scale['euc'].unsqueeze(-1).square()
            else:
                raise NotImplementedError()

        if self.rot_num:

            def est_score(data_mats, est_rots, scale):
                rel_mat = torch.einsum(
                    'nkmab,nkmcb->nkmac', data_mats, est_rots)
                rel_aa = pttf.matrix_to_axis_angle(rel_mat)
                return self.so3_noise.score_vec(
                    vec=rel_aa, scale=scale['rot'].unsqueeze(-1).repeat(1, 1, self.rot_num))

            B, N = len(tensor), self.rot_num
            if self.config.score_rot_rep == 'aa':
                raise NotImplementedError()
                result_dict['rot'] = tensor[:, self.euc_dim:].reshape(B, N, 3)
            elif self.config.score_euc_rep == 'epsilon':
                result_dict['rot'] = (tensor[:, self.euc_dim:] / scale['rot'][:, None]).reshape(B, N, 3)
            elif self.config.score_rot_rep == 'rel_mat':
                raise NotImplementedError()
                mats = tensor[..., self.euc_dim:].reshape(B, N, 3, 3)
                data_mats = dic['rot']
                new_mat = mats @ data_mats.transpose(-1, -2)
                result_dict['rot'] = torch.cat([new_mat[..., 2, [1]]-new_mat[..., 1, [2]], new_mat[..., 0, [2]]-new_mat[..., 2, [
                                               0]], new_mat[..., 1, [0]] - new_mat[..., 0, [1]]], dim=-1).reshape(B, N, 3) / 2
            elif self.config.score_rot_rep == 'est_4d':
                raise NotImplementedError()
                est_quats = tensor[..., self.euc_dim:].reshape(B, N, 4)
                est_norm_quats = est_quats / est_quats.norm(dim=-1, keepdim=True)
                est_rots = pttf.quaternion_to_matrix(est_norm_quats)
                data_mats = dic['rot']
                result_dict['rot'] = est_score(data_mats, est_rots, scale)
            elif self.config.score_rot_rep == 'est_9d':
                raise NotImplementedError()
                est_mats = tensor[..., self.euc_dim:].reshape(B, N, 3, 3)
                U, _, V = torch.svd(est_mats)
                est_rots = torch.einsum('nkmab,nkmcb->nkmac', U, V)
                data_mats = dic['rot']
                result_dict['rot'] = est_score(data_mats, est_rots, scale)

        return result_dict

    # total dimension of representation
    def rep_dim(self):
        return self.euc_dim + self.rot_num * 9

    # flatten rep dic into tensor(..., D+9N)
    def rep_to_tensor(self, dic):
        if not self.euc_dim:
            return dic['rot'].reshape(*dic['rot'].shape[:-3], 9*self.rot_num)
        elif not self.rot_num:
            return dic['euc']

        euc = dic['euc']
        rot = dic['rot'].reshape(*dic['rot'].shape[:-3], 9*self.rot_num)
        return torch.cat([euc, rot], dim=-1)

    # pack rep tensor(..., D+9N) into dict
    def rep_to_dict(self, tensor):
        result_dict = {}
        if self.euc_dim:
            result_dict['euc'] = tensor[..., :self.euc_dim]
        if self.rot_num:
            raw_rot = tensor[..., self.euc_dim:].reshape(
                *tensor.shape[:-1], -1, 3, 3)
            rot_shape = raw_rot.shape
            U, S, V = torch.svd(raw_rot.reshape(-1, 3, 3))
            rot = torch.einsum('nab,nbc->nac', U, V.transpose(-1, -2))
            result_dict['rot'] = rot.reshape(*rot_shape)
        return result_dict

    # scale: (B,) -> noise (B, D/N, 3, 3)
    def start_noise(self, scale):
        new_dict = {}

        if self.euc_dim:
            new_dict['euc'] = torch.randn((len(scale['euc']), self.euc_dim), device=scale['euc'].device)
            log_prob = (-0.5 * (new_dict['euc'] / scale['euc'].unsqueeze(-1)).square() - torch.log(scale['euc'] * np.sqrt(2 * np.pi)).unsqueeze(-1)).sum(dim=-1)

        if self.rot_num:
            if log_prob is None:
                log_prob = torch.zeros_like(scale['rot'])
            rot_num = len(scale['rot']) * self.rot_num
            new_dict['rot'] = pttf.random_rotations(rot_num, device=scale['rot'].device).reshape(-1, self.rot_num, 3, 3)

        new_dict['log_prob'] = log_prob
        return new_dict

    def get_shape(self, dic):
        if self.euc_dim:
            return dic['euc'].shape[:-1]
        else:
            return dic['rot'].shape[:-3]

    # data (B, D/N, 3, 3) -> dx (B, D + 3N)
    def get_dx(self, dic, device, dtype):

        dx = torch.zeros((*self.get_shape(dic), self.score_dim()), device=device, dtype=dtype)
        dx.requires_grad_(self.config.ode and self.config.log_prob_type is not None)

        if self.euc_dim:
            dic['euc'] += dx[..., :self.euc_dim]

        if self.rot_num:
            dx_aa = dx[..., self.euc_dim:].reshape(-1, 3)
            dx_mat = pttf.axis_angle_to_matrix(dx_aa)
            rot = dic['rot'].reshape(-1, 3, 3)
            new_rot = torch.einsum('nab,nbc->nac', rot, dx_mat)
            dic['rot'] = new_rot.reshape(*dic['rot'].shape)

        return dic, dx

    def add_noise(self, dic: dict, scale: torch.Tensor):
        """
            add noise for euc & rot with specified std(scale)
            dic: euc/rot (B, D/N, 3, 3), scale: (B,)
            return euc/rot score (B, D/N, 3)
        """
        noise_dic = {}
        score_dic = {}

        if self.euc_dim:
            euc, euc_scale = dic['euc'], scale['euc'].unsqueeze(-1)
            noise_euc = torch.randn_like(euc)
            noise_dic['euc'] = noise_euc * euc_scale + euc * (1-euc_scale.square()).sqrt()
            score_dic['euc'] = -noise_euc / euc_scale

        if self.rot_num:
            rot = dic['rot']
            B, N, _, _ = rot.shape
            rot = rot.reshape(-1, 3, 3)  # (B * N, 3, 3)
            repeat_scale = repeat(scale['rot'], 'b -> b n', n=N)  # (B, N)
            aa_noise = self.so3_noise.sample(repeat_scale)  # (B, N, 3)
            m_noise = pttf.axis_angle_to_matrix(aa_noise.reshape(-1, 3))  # (B * N, 3, 3)
            noise_rot = torch.einsum('nab,nbc->nac', rot, m_noise)  # (B * N, 3, 3)
            noise_dic['rot'] = noise_rot.reshape(B, N, 3, 3)  # (B, N, 3, 3)
            score_dic['rot'] = -self.so3_noise.score_vec(vec=aa_noise, scale=repeat_scale)

        return noise_dic, score_dic

    # score: (B, D/N, 3), scale: (B,) -> new score (B, D/N, 3)
    # def scale_score(self, dic, scale):
    #     if self.config.score_rot_rep == 'est_4d' or self.config.score_rot_rep == 'est_9d':
    #         return dic

    #     new_dic = {}

    #     if self.euc_dim:
    #         new_dic['euc'] = dic['euc'] / \
    #             scale['euc'].reshape(*dic['euc'].shape[:-1], 1)

    #     if self.rot_num:
    #         if self.config.scale_rot_score_norm:
    #             new_dic['rot'] = dic['rot'] * self.so3_noise.score_norm(
    #                 scale['rot']).reshape(*dic['rot'].shape[:-2], 1, 1)
    #         else:
    #             new_dic['rot'] = dic['rot'] / \
    #                 scale['rot'].reshape(*dic['rot'].shape[:-2], 1, 1)

    #     return new_dic

    def direct_loss(self, gt_direct, est_direct):
        """
            compute loss according to the estimation
            L1 loss for euc, L2 loss for rot

            gt_direct: dict with euc/rot (B, D/N, 3, 3)
            est_direct: dict with euc/rot (B, D/N, 3, 3)
            return loss (B, D/N)
        """
        loss = {}
        if self.euc_dim:
            loss['euc'] = (gt_direct['euc'] - est_direct['euc']).abs()
        if self.rot_num:
            loss['rot'] = (gt_direct['rot'] - est_direct['rot']).square().mean(dim=(-1, -2))
        return loss

    def score_loss(self, scale_dic, gt_score, est_score):
        """
            compute loss according to the score

            scale_dic: dict with euc/rot (B,)
            gt_score: dict with euc/rot (B, D/N, 3)
            est_score: dict with euc/rot (B, D/N, 3)
            return loss (B, D/N, 3)
        """
        loss_score_dic = {}

        if self.euc_dim:
            euc_score, euc_est_score = gt_score['euc'], est_score['euc']
            euc_scale = scale_dic['euc'][:, None]
            euc_loss = (euc_score - euc_est_score) ** 2
            # if self.config.scale_loss == 'direct' or self.config.scale_loss == 'exp':
                # euc_loss *= euc_scale ** 2
            # elif self.config.scale_loss == 'min_kl':
                # euc_loss *= 2 * euc_scale / (1 - euc_scale.square())
            if self.config.scale_loss == 'epsilon':
                euc_loss *= euc_scale ** 2
            loss_score_dic['euc'] = euc_loss

        if self.rot_num:
            rot_score, rot_est_score = gt_score['rot'], est_score['rot']
            rot_scale = scale_dic['rot'][:, None, None]
            rot_score_norm = self.so3_noise.score_norm(rot_scale)
            rot_loss = (rot_score - rot_est_score) ** 2
            # if self.config.scale_loss == 'direct':
                # rot_loss *= scale_dic['rot'].reshape(*
                                                    #  rot_shape[:-2], 1, 1) ** 2
            # elif self.config.scale_loss == 'exp':
                # rot_loss /= rot_score_norm ** 2
            # elif self.config.scale_loss == 'min_kl':
                # rot_loss *= scale_dic['rot'].reshape(*rot_shape[:-2], 1, 1)
            if self.config.scale_loss == 'epsilon':
                rot_loss *= rot_scale ** 2
            elif self.config.scale_loss == 'norm':
                rot_loss /= rot_score_norm ** 2
            loss_score_dic['rot'] = rot_loss

        return loss_score_dic

    # data: (B, D/N, 3, 3)
    # score: (B, D/N, 3)
    # scale: (B,)
    # dt: scalar
    # no_noise: bool
    # dx: (B, D + 3N)
    # return new data: (B, D/N, 3, 3)
    def score_update(self, data, score, t, dt, no_noise, dx):
        new_dict = {}
        dys = []

        if self.euc_dim:
            dlog_alpha_prod = self.t_schedule.dlog_alpha_prod(t[:, None])
            beta = -dlog_alpha_prod
            f = -1/2 * beta * data['euc']
            g = beta.sqrt()
            if no_noise or self.config.ode:
                euc_perturb = f * dt - g.square() * score['euc'] * dt / 2
            else:
                euc_perturb = f * dt - g.square() * score['euc'] * dt + g * math.sqrt(abs(dt)) * torch.randn_like(data['euc'])
            dys.append(euc_perturb / dt)
            new_dict['euc'] = data['euc'] + euc_perturb

        if self.rot_num:
            rot_g = self.t_schedule.sqrt_d_scale_2(t[:, None, None])
            rot_z = torch.randn_like(score['rot'])
            if no_noise or self.config.ode:
                aa_rot_perturb = score['rot'] * dt * rot_g ** 2 / 2
            else:
                aa_rot_perturb = score['rot'] * dt * rot_g ** 2 + rot_g * math.sqrt(abs(dt)) * rot_z
            m_rot_perturb = pttf.axis_angle_to_matrix(aa_rot_perturb.reshape(-1, 3))
            dys.append(rearrange(aa_rot_perturb / dt,  'b n d -> b (n d)'))
            new_dict['rot'] = torch.einsum('nkab,nkbc->nkac', data['rot'], m_rot_perturb.reshape(*data['rot'].shape))
        
        dy = torch.cat(dys, dim=-1)
        if self.config.log_prob_type is not None:
            new_dict['log_prob'] = data['log_prob'] - dt * self.jacobian_trace(dy, dx)

        return new_dict

    def jacobian_trace(self, dy, dx):
        return jacobian_trace(self.config.log_prob, dx, dy)

    def get_score_norm(self, score):
        raise NotImplementedError()
        # score_list = []
        # if self.euc_dim:
        #     score_list.append(score['euc'])
        # if self.rot_num:
        #     score_list.append(score['rot'].reshape(
        #         *score['rot'].shape[:-2], -1))
        # return torch.cat(score_list, dim=-1).norm(dim=-1, keepdim=True)

    def change_score_norm(self, score, old_norm, new_norm):
        raise NotImplementedError()
        # new_dict = {}
        # if self.euc_dim:
        #     new_dict['euc'] = score['euc'] / old_norm * new_norm
        # if self.rot_num:
        #     new_dict['rot'] = score['rot'] / \
        #         old_norm.unsqueeze(-1) * new_norm.unsqueeze(-1)
        # return new_dict