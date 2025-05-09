import math
import numpy as np
import torch

from .distributions.so3_utils import MAX_EPS, EPS_TO_SCALE

def get_tschedule(config: dict):
    if config.schedule.type == 'cosine':
        return TSchedule_cosine(config.schedule)
    elif config.schedule.type == 'linear':
        return TSchedule_linear(config.schedule)
    elif config.schedule.type == 'exp':
        return TSchedule_exp(config.schedule)

class TSchedule:
    def __init__(self, config: dict):
        self.config = config
        self.rot_scale = MAX_EPS * EPS_TO_SCALE
        self.rot_bias = 0
    
    def scale(self, t: torch.Tensor):
        """
            x_t = sqrt(1 - scale ^ 2) * x_0 + scale * z
        """
        pass

    def alpha_prod(self, t: torch.Tensor):
        """
            x_t = sqrt(alpha_prod(t)) * x_0 + sqrt(1 - alpha_prod(t)) * z
        """
        pass

    def dlog_alpha_prod(self, t: torch.Tensor):
        """
            d log(alpha_prod(t)) / dt
        """
        pass

    def sqrt_d_scale_2(self, t: torch.Tensor):
        """
            sqrt(d scale^2 / dt)
        """
        pass
    
    def t_to_scale(self, t: torch.Tensor):
        # alpha_prod = self.alpha_prod(t)
        # euc_scale = (1-alpha_prod).sqrt()
        euc_scale = self.scale(t)
        rot_scale = euc_scale * self.rot_scale + self.rot_bias
        return dict(euc=euc_scale, rot=rot_scale)

    def get_train_t(self, batch_size, device, dtype):
        t = torch.rand((batch_size,), device=device, dtype=dtype)
        # t = torch.rand((batch_size,), device=device, dtype=dtype) * 0.998 + 0.001
        return t

class TSchedule_cosine(TSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        s = config.s
        M = MAX_EPS * EPS_TO_SCALE
        self.rot_scale = (M - s) / (1 - s)
        self.rot_bias = (-M + 1) * s / (1 - s)
    
    def scale(self, t: torch.Tensor):
        return (1 - self.alpha_prod(t)).sqrt()
    
    def alpha_prod(self, t: torch.Tensor):
        s = self.config.s
        return torch.cos(torch.pi/2*(t+s)/(1+s)).square()

    def dlog_alpha_prod(self, t: torch.Tensor):
        s = self.config.s
        return -torch.pi/(1+s)*torch.tan(torch.pi/2*(t+s)/(1+s))

    def sqrt_d_scale_2(self, t: torch.Tensor):
        s = self.config.s
        k, b, pi = self.rot_scale, self.rot_bias, torch.pi
        theta = pi*(t+s)/(1+s)/2
        return torch.sqrt(k*pi/(1+s)*torch.cos(theta)*(k*torch.sin(theta)+b))

class TSchedule_exp(TSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.M, self.m = config.max, config.min
    
    def scale(self, t: torch.Tensor):
        return self.m * ((self.M / self.m) ** t)

    def alpha_prod(self, t: torch.Tensor):
        return 1 - (self.m ** 2) * ((self.M / self.m) ** (2 * t))

    def dlog_alpha_prod(self, t: torch.Tensor):
        return -2 * math.log(self.M / self.m) *  (1 - self.alpha_prod(t)) / self.alpha_prod(t)

    def sqrt_d_scale_2(self, t: torch.Tensor):
        return self.rot_scale * self.m * ((self.M / self.m) ** t) * math.sqrt(2 * math.log(self.M / self.m))
    
class TSchedule_linear(TSchedule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.M, self.m = config.max, config.min
    
    def scale(self, t: torch.Tensor):
        return self.m + t * (self.M - self.m)

    def dlog_alpha_prod(self, t: torch.Tensor):
        return -2 * self.scale(t) * (self.M - self.m) / (1 - self.scale(t) ** 2)

    def sqrt_d_scale_2(self, t: torch.Tensor):
        return self.rot_scale * (2 * self.scale(t) * (self.M - self.m)).sqrt()