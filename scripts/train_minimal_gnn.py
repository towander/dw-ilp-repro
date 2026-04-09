# scripts/train_minimal_gnn.py

import os
import pickle
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    def __init__(
        self,
        task_dim=5,
        server_dim=5,
        pair_dim=5,
        hidden_dim=64,
        emb_dim=32,
        dropout=0.0,
    ):
        super().__init__()
        self.task_encoder = MLP(task_dim, hidden_dim, emb_dim, dropout)
        self.server_encoder = MLP(server_dim, hidden_dim, emb_dim, dropout)
        self.pair_encoder = MLP(pair_dim, hidden_dim, emb_dim, dropout)

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

        self.score_head = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, task_feat, server_feat, pair_feat, mask):
        T = task_feat.size(0)
        S = server_feat.size(0)

        t_emb = self.task_encoder(task_feat)
        s_emb = self.server_encoder(server_feat)
        p_emb = self.pair_encoder(pair_feat)

        valid = mask.float().unsqueeze(-1)

        s_expand = s_emb.unsqueeze(0).expand(T, S, -1)
        t_expand = t_emb.unsqueeze(1).expand(T, S, -1)

        task_msgs = (s_expand + p_emb) * valid
        task_den = valid.sum(dim=1).clamp_min(1.0)
        task_aggr = task_msgs.sum(dim=1) / task_den
        t_emb2 = self.task_update(torch.cat([t_emb, task_aggr], dim=-1))

        server_msgs = (t_expand + p_emb) * valid
        server_den = valid.sum(dim=0).clamp_min(1.0)
        server_aggr = server_msgs.sum(dim=0) / server_den
        s_emb2 = self.server_update(torch.cat([s_emb, server_aggr], dim=-1))

        t2_expand = t_emb2.unsqueeze(1).expand(T, S, -1)
        s2_expand = s_emb2.unsqueeze(0).expand(T, S, -1)

        fused = torch.cat([t2_expand, s2_expand, p_emb], dim=-1)
        logits = self.score_head(fused).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits


def masked_bce_loss(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    loss = loss * mask.float()
    return loss.sum() / (mask.float().sum() + 1e-8)


def load_samples(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_tensor_sample(sample, device):
    return {
        "task_features": torch.tensor(sample["task_features"], dtype=torch.float32, device=device),
        "server_features": torch.tensor(sample["server_features"], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(sample["pair_features"], dtype=torch.float32, device=device),
        "mask": torch.tensor(sample["mask"], dtype=torch.bool, device=device),
        "labels": torch.tensor(sample["labels"], dtype=torch.float32, device=device),
    }


def evaluate(model, samples, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sample in samples:
            batch = to_tensor_sample(sample, device)
            logits = model(
                batch["task_features"],
                batch["server_features"],
                batch["pair_features"],
                batch["mask"],
            )
            loss = masked_bce_loss(logits, batch["labels"], batch["mask"])
            total_loss += loss.item()
    return total_loss / max(len(samples), 1)


def main():
    with open("config/minimal_gnn.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"].get("seed", 42))
    device = cfg["train"].get("device", "cpu")

    train_samples = load_samples(cfg["paths"]["train_data"])
    val_samples = load_samples(cfg["paths"]["val_data"])

    model = MinimalBipartiteGNN(
        hidden_dim=cfg["model"].get("hidden_dim", 64),
        emb_dim=cfg["model"].get("emb_dim", 32),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["train"].get("lr", 1e-3))
    best_val = float("inf")

    for epoch in range(1, cfg["train"].get("epochs", 20) + 1):
        model.train()
        random.shuffle(train_samples)

        total_loss = 0.0
        for sample in train_samples:
            batch = to_tensor_sample(sample, device)
            logits = model(
                batch["task_features"],
                batch["server_features"],
                batch["pair_features"],
                batch["mask"],
            )
            loss = masked_bce_loss(logits, batch["labels"], batch["mask"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(len(train_samples), 1)
        val_loss = evaluate(model, val_samples, device)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = cfg["paths"]["checkpoint"]
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"[MinimalGNN] saved best checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
