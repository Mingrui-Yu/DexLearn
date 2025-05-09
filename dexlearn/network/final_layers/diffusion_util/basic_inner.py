import torch
from torch import nn

from .manifold import Manifold
from .naive_diffusion import SinusoidalPosEmb
from ..mlp import BasicMLP


class BasicInner(nn.Module):
    """
    Inner model required by diffusion
    """

    def __init__(
        self,
        config: dict,
        diffusion_manifold: Manifold,
        direct_manifold: Manifold,
    ):
        super().__init__()
        self.config = config
        self.diffusion_manifold = diffusion_manifold
        self.direct_manifold = direct_manifold
        self.score_rep_dim = diffusion_manifold.score_rep_dim()
        self.direct_est_dim = direct_manifold.rep_dim()
        self.sin_embedding = SinusoidalPosEmb(config.t_dim, theta=1000)
        self.mlp = BasicMLP(
            config.feature_dim + diffusion_manifold.rep_dim(), self.score_rep_dim
        )

    def forward(self, mode: str, *args, **kwargs):
        """
        mode: "score" or "direct"
        type of args and kwargs can be found in the following functions
        """
        if mode == "score":
            return self.score(*args, **kwargs)
        elif mode == "direct":
            return self.direct(*args, **kwargs)

    def score(self, cur: dict, t: torch.Tensor, scale: torch.Tensor, cond: dict):
        """
        Compute score given the input

        cur: dict with rot (B, M, 3, 3), euc (B, N)
        t: torch.Tensor (B,)
        scale: torch.Tensor (B,)
        cond: conditional input

        returns dict with rot score (B, M, ...), euc (B, N)
        """
        t_embed = self.sin_embedding(t)
        x = torch.cat(
            [self.diffusion_manifold.rep_to_tensor(cur), cond["feat"] + t_embed], dim=-1
        )
        score = self.mlp(x)
        return self.diffusion_manifold.score_to_dict(score, cur, scale)

    def direct(self, cur: dict, cond: dict):
        """
        Compute the part that is directly estimated

        cur: dict with rot (B, M, 3, 3), euc (B, N)
        cond: conditional input
        """
        raise NotImplementedError
