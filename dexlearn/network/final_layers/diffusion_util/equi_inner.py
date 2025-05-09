import torch
from einops import repeat

from .basic_inner import BasicInner
from .manifold import Manifold
from ..mlp import BasicMLP
from ...backbones.pointnet import RelPointnet


class EquiInner(BasicInner):
    def __init__(
        self, config: dict, diffusion_manifold: Manifold, direct_manifold: Manifold
    ):
        super().__init__(config, diffusion_manifold, direct_manifold)
        self.pn = RelPointnet(
            point_feature_dim=3 + config.t_dim, pc_feature_dim=config.feature_dim
        )
        self.mlp = BasicMLP(
            config.feature_dim + 12, self.score_rep_dim + self.direct_est_dim
        )
        self.register_buffer("trans_mean", torch.tensor(config.trans_mean))
        self.register_buffer("trans_std", torch.tensor(config.trans_std))

    def est(self, cur: dict, t: torch.Tensor, cond: dict):
        """
        Compute score and direct estimation jointly

        cur: dict with rot (B, 1, 3, 3), euc (B, N)
            assume translation is the first three elements
        t: torch.Tensor (B,)
        cond: dict with pc torch.Tensor (B, K, 3)

        return tensor with shape (B, D)
        """
        pc = cond["pc"]
        emb = self.sin_embedding(t)
        pc = torch.cat([pc, repeat(emb, "b d -> b k d", k=pc.shape[1])], dim=-1)

        rot = cur["rot"][:, 0]
        trans = cur["euc"][:, :3] * self.trans_std + self.trans_mean

        if self.config.equi:
            feat, _ = self.pn(pc, rot, trans)
        else:
            feat, _ = self.pn(
                pc, torch.eye(3).cuda() + torch.zeros_like(rot), torch.zeros_like(trans)
            )

        result = self.mlp(torch.cat([feat, rot.reshape(-1, 9), trans], dim=-1))
        if self.config.equi:
            trans_score = torch.einsum("nab,nb->na", rot, result[:, :3])
            result = torch.cat([trans_score, result[:, 3:]], dim=-1)
        return result

    def score(self, cur: dict, t: torch.Tensor, scale: torch.Tensor, cond: dict):
        """
        Compute score given the input

        cur: dict with rot (B, 1, 3, 3), euc (B, N)
        t: torch.Tensor (B,)
        cond: dict with pc torch.Tensor (B, K, 3)

        returns dict with rot score (B, 1, ...), euc (B, N)
        """
        result = self.est(cur, t, cond)
        score = result[:, : self.score_rep_dim]
        return self.diffusion_manifold.score_to_dict(score, cur, scale)

    def direct(self, cur: dict, cond: dict):
        """
        Compute the part that is directly estimated

        cur: dict with rot (B, 1, 3, 3), euc (B, N)
        cond: dict with pc torch.Tensor (B, K, 3)

        return
        """
        device, dtype = cond["pc"].device, cond["pc"].dtype
        t = torch.full((cond["pc"].shape[0],), -1, device=device, dtype=dtype)
        result = self.est(cur, t, cond)
        return self.direct_manifold.rep_to_dict(result[:, self.score_rep_dim :])
