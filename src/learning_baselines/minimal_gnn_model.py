# src/learning_baselines/minimal_gnn_model.py

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MinimalBipartiteGNN(nn.Module):
    """
    低开销、轻量化 GNN baseline

    输入:
        task_feat:   [T, Dt]
        server_feat: [S, Ds]
        pair_feat:   [T, S, Dp]
        mask:        [T, S]  bool, True=可行

    输出:
        logits:      [T, S]
    """

    def __init__(
        self,
        task_dim: int,
        server_dim: int,
        pair_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.task_encoder = MLP(task_dim, hidden_dim, emb_dim, dropout)
        self.server_encoder = MLP(server_dim, hidden_dim, emb_dim, dropout)
        self.pair_encoder = MLP(pair_dim, hidden_dim, emb_dim, dropout)

        # 一轮轻量 message passing
        self.task_update = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

        self.server_update = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
        )

        # pair 打分头
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, task_feat, server_feat, pair_feat, mask):
        """
        task_feat:   [T, Dt]
        server_feat: [S, Ds]
        pair_feat:   [T, S, Dp]
        mask:        [T, S] bool
        """
        T = task_feat.size(0)
        S = server_feat.size(0)

        t_emb = self.task_encoder(task_feat)         # [T, E]
        s_emb = self.server_encoder(server_feat)     # [S, E]
        p_emb = self.pair_encoder(pair_feat)         # [T, S, E]

        valid = mask.float().unsqueeze(-1)           # [T, S, 1]

        s_expand = s_emb.unsqueeze(0).expand(T, S, -1)   # [T, S, E]
        t_expand = t_emb.unsqueeze(1).expand(T, S, -1)   # [T, S, E]

        # task 从 server 聚合
        task_msgs = (s_expand + p_emb) * valid
        task_den = valid.sum(dim=1).clamp_min(1.0)       # [T, 1]
        task_aggr = task_msgs.sum(dim=1) / task_den      # [T, E]
        t_emb2 = self.task_update(torch.cat([t_emb, task_aggr], dim=-1))

        # server 从 task 聚合
        server_msgs = (t_expand + p_emb) * valid
        server_den = valid.sum(dim=0).clamp_min(1.0)     # [S, 1]
        server_aggr = server_msgs.sum(dim=0) / server_den
        s_emb2 = self.server_update(torch.cat([s_emb, server_aggr], dim=-1))

        # pair score
        t2_expand = t_emb2.unsqueeze(1).expand(T, S, -1)
        s2_expand = s_emb2.unsqueeze(0).expand(T, S, -1)

        fused = torch.cat([t2_expand, s2_expand, p_emb], dim=-1)
        logits = self.score_head(fused).squeeze(-1)      # [T, S]

        # 不可行位置置极小值，避免后续选中
        logits = logits.masked_fill(~mask, -1e9)
        return logits
