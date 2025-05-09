import numpy as np
import torch
from torch import nn
from tqdm import trange
from typing import Callable
from pytorch3d import transforms as pttf
from einops import repeat

from .t_schedule import get_tschedule
from .manifold import Manifold
from .distributions.so3_utils import _expansion, EPS_TO_SCALE

class DiffusionModel(nn.Module):

    def __init__(self, config: dict, model: Callable, diffusion_manifold: Manifold, direct_manifold: Manifold) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.diffusion_manifold = diffusion_manifold
        self.direct_manifold = direct_manifold
        self.t_schedule = get_tschedule(config)

        # self.self_test()
    
    def forward(self, data: dict, mode: str):
        """
            mode: train or eval
            see _train and _sample for more details
        """
        if mode == "train":
            return self._train(data)
        elif mode == "eval":
            return self._sample(data)
        else:
            raise NotImplementedError()

    def _train(self, data: dict):
        """
            batch_size: int
            device: torch.device
            dtype: torch.dtype
            diffusion: dict(rot: torch.Tensor (B, K1, 3, 3), euc: torch.Tensor, (B, K2))
            direct: dict(rot: torch.Tensor (B, K3, 3, 3), euc: torch.Tensor, (B, K4))
            cond: dict
        """
        batch_size, device, dtype = data['batch_size'], data['device'], data['dtype']
        sample_num = self.config.train_sample_num
        for k1 in ['diffusion', 'direct']:
            if k1 not in data:
                continue
            for k2 in ['rot', 'euc']:
                data[k1][k2] = data[k1][k2].repeat_interleave(sample_num, 0)
        data['cond']['feat'] = data['cond']['feat'].repeat_interleave(sample_num, 0)
        batch_size *= sample_num

        t = self.t_schedule.get_train_t(batch_size, device, dtype)
        scale = self.t_schedule.t_to_scale(t)
        noised_input, gt_score = self.diffusion_manifold.add_noise(data['diffusion'], scale)

        est_score = self.model('score', noised_input, t, scale, data['cond'])

        data['score_loss'] = self.diffusion_manifold.score_loss(scale, est_score, gt_score)
        if self.direct_manifold.euc_dim or self.direct_manifold.rot_num:
            est_direct = self.model('direct', noised_input, data['cond'])
            data['direct_loss'] = self.direct_manifold.direct_loss(est_direct, data['direct'])
        return data

    def _sample(self, data: dict):
        """
            batch_size: int
            device: torch.device
            dtype: torch.dtype
            cond: dict
        """
        batch_size, device, dtype = data['batch_size'], data['device'], data['dtype']

        t = torch.full((batch_size,), 0.999, device=device, dtype=dtype)
        scale = self.t_schedule.t_to_scale(t)
        est_diffusion = self.diffusion_manifold.start_noise(scale)
        data['est_diffusion_store'] = [{k: v.cpu() for k, v in est_diffusion.items()}]

        dt = 1 / self.config.inference_steps
        with torch.set_grad_enabled(self.config.log_prob_type is not None):
            for i in range(self.config.inference_steps-1, 0, -1):
                t = torch.full_like(t, i * dt)
                scale = self.t_schedule.t_to_scale(t)
                est_diffusion, dx = self.diffusion_manifold.get_dx(est_diffusion, device, dtype)
                est_score = self.model('score', est_diffusion, t, scale, data['cond'])
                no_noise = (i == 1) and self.config.no_final_step_noise
                est_diffusion = self.diffusion_manifold.score_update(est_diffusion, est_score, t, -dt, no_noise, dx)
                est_diffusion = {k: v.detach() for k, v in est_diffusion.items()}
                data['est_diffusion_store'].append({k: v.cpu() for k, v in est_diffusion.items()})

        if self.direct_manifold.euc_dim or self.direct_manifold.rot_num:
            data['est_direct'] = self.model('direct', est_diffusion, data['cond'])
        data['est_diffusion'] = est_diffusion
        return data
    
    def self_test(self):
        """
            Test diffusion's result with ground truth score
        """
        assert self.diffusion_manifold.euc_dim == 3 and self.diffusion_manifold.rot_num == 1
        batch_size, device, dtype = 5, torch.device('cuda'), torch.float32
        store = []

        # gt_euc = torch.tensor([[0,0,0]], device=device, dtype=dtype)
        gt_euc = torch.tensor([[0,0,0],[1,1,1]], device=device, dtype=dtype)
        gt_rot = torch.tensor([[[1,0,0],[0,0,1],[0,-1,0]], [[1,0,0],[0,1,0],[0,0,1]]], device=device, dtype=dtype)

        t = torch.ones((batch_size,), device=device, dtype=dtype)
        scale = self.t_schedule.t_to_scale(t)
        est_diffusion = self.diffusion_manifold.start_noise(scale)
        store = [{k: v.cpu() for k, v in est_diffusion.items()}]

        # self.config.ode=0
        self.config.inference_steps = 100
        dt = 1 / self.config.inference_steps
        for i in trange(self.config.inference_steps-1, 0, -1):
            # if i == 100:
                # print("!")
            t = torch.full_like(t, i * dt)
            scale = self.t_schedule.t_to_scale(t)

            score = dict()
            std_noise = (est_diffusion['euc'] - gt_euc[:, None] * (1 - scale['euc'][:, None].square()).sqrt()) / scale['euc'][:, None]
            # print(i*dt, scale['euc'][0].cpu(), est_diffusion['euc'][0].cpu(), std_noise[:,0].cpu())
            prob = (-(est_diffusion['euc'] - gt_euc[:, None]).square().sum(-1) / 2 / scale['euc'].square()).exp()
            std_noise = ((std_noise * prob[:,:,None]).sum(0) / prob[:,:,None].sum(0))
            score['euc'] = (-std_noise / scale['euc'][:, None])
            m_noise = torch.einsum('mba,nkbc->mnkac', gt_rot, est_diffusion['rot'])
            aa_noise = pttf.matrix_to_axis_angle(m_noise.reshape(-1, 3, 3))
            aa_noise_norm = aa_noise.norm(dim=-1, keepdim=True)
            aa_noise = torch.where(aa_noise_norm > torch.pi, -aa_noise/aa_noise_norm*(torch.pi*2-aa_noise_norm), aa_noise)[:, None]
            # prob = None
            prob = torch.from_numpy(np.concatenate([_expansion(aa_noise_norm[i].cpu().numpy(), scale['rot'][:1].cpu().numpy()/EPS_TO_SCALE) for i in range(len(aa_noise))])).to(device).to(dtype).reshape(len(gt_rot), -1)
            score_rot = -self.diffusion_manifold.so3_noise.score_vec(repeat(scale['rot'], 'b -> (m b) 1', m=len(gt_rot)), aa_noise).reshape(len(gt_rot), -1, 1, 3)
            score['rot'] = (score_rot * prob[:,:,None,None]).sum(0) / prob[:,:,None,None].sum(0)
            # score['rot'] = score_rot.mean(0)

            no_noise = (i == 1) and self.config.no_final_step_noise
            est_diffusion = self.diffusion_manifold.score_update(est_diffusion, score, t, -dt, no_noise)
            # alpha = (self.t_schedule.alpha_prod(t)/self.t_schedule.alpha_prod(t-dt))[:, None]
            # est_diffusion['euc'] = 1/alpha.sqrt()*est_diffusion['euc'] - (1-alpha) / scale['euc'][:,None] / alpha.sqrt() * std_noise + torch.randn_like(est_diffusion['euc']) * ((1-alpha) * (1-self.t_schedule.alpha_prod(t-dt)[:, None]) / (1-self.t_schedule.alpha_prod(t))[:, None]).sqrt()
            store.append({k: v.cpu() for k, v in est_diffusion.items()})

        print(est_diffusion['euc'])
        print(est_diffusion['rot'])
        return est_diffusion, store