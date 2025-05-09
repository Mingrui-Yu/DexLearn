import torch


class LearnableTypeCond(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grasp_type_feat = torch.nn.Embedding(
            num_embeddings=40, embedding_dim=cfg.out_feat_dim
        )
        return

    def forward(self, data):
        if self.cfg.disabled:
            return self.grasp_type_feat(data["grasp_type_id"] * 0)
        else:
            return self.grasp_type_feat(data["grasp_type_id"])
