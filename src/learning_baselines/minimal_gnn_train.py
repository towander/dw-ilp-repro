# src/learning_baselines/minimal_gnn_train.py

import os
import random
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# =========================
# common utils
# =========================

def safe_get(obj, name, default=0.0):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# feature builders
# =========================

def build_task_features(tasks, current_time):
    """
    输出: [T, 5]
    5维:
      0 required_compute
      1 input_size
      2 slack_time
      3 priority
      4 waiting_time
    """
    feats = []
    for t in tasks:
        required_compute = float(safe_get(t, "required_compute", safe_get(t, "flops", 0.0)))
        input_size = float(safe_get(t, "input_size", safe_get(t, "data_size", 0.0)))
        deadline = float(safe_get(t, "deadline", current_time + 100.0))
        priority = float(safe_get(t, "priority", 1.0))
        arrival_time = float(safe_get(t, "arrival_time", current_time))

        slack_time = max(0.0, deadline - current_time)
        waiting_time = max(0.0, current_time - arrival_time)

        feats.append([
            required_compute,
            input_size,
            slack_time,
            priority,
            waiting_time,
        ])

    if len(feats) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def build_server_features(servers):
    """
    输出: [S, 5]
    5维:
      0 compute_capacity
      1 parallelism
      2 queue_length
      3 remaining_energy
      4 battery_ratio
    """
    feats = []
    for s in servers:
        compute_capacity = float(safe_get(s, "compute_capacity", safe_get(s, "cpu_capacity", 1.0)))
        parallelism = float(safe_get(s, "parallelism", 1.0))
        queue_length = float(safe_get(s, "queue_length", 0.0))
        remaining_energy = float(safe_get(s, "remaining_energy", safe_get(s, "energy", 100.0)))
        battery_ratio = float(safe_get(s, "battery_ratio", 1.0))

        feats.append([
            compute_capacity,
            parallelism,
            queue_length,
            remaining_energy,
            battery_ratio,
        ])

    if len(feats) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


def estimate_pair_values(task, server, current_time):
    """
    轻量启发式估计。
    这部分和你工具分析里出现的逻辑一致风格：
    - tx_delay = input_size / 10
    - exec_time = required_compute / compute_capacity
    - queue_delay = queue_length * 0.5
    - energy_cost = 0.05 * required_compute + 0.01 * input_size
    """
    required_compute = float(safe_get(task, "required_compute", safe_get(task, "flops", 1.0)))
    input_size = float(safe_get(task, "input_size", safe_get(task, "data_size", 1.0)))
    deadline = float(safe_get(task, "deadline", current_time + 100.0))

    compute_capacity = float(safe_get(server, "compute_capacity", safe_get(server, "cpu_capacity", 1.0)))
    queue_length = float(safe_get(server, "queue_length", 0.0))
    remaining_energy = float(safe_get(server, "remaining_energy", safe_get(server, "energy", 100.0)))

    tx_delay = input_size / 10.0
    exec_time = required_compute / max(compute_capacity, 1e-6)
    queue_delay = queue_length * 0.5
    total_latency = tx_delay + exec_time + queue_delay
    deadline_margin = deadline - (current_time + total_latency)
    energy_cost = 0.05 * required_compute + 0.01 * input_size

    feasible = (deadline_margin >= 0.0) and (remaining_energy >= energy_cost)
    return tx_delay, exec_time, total_latency, deadline_margin, energy_cost, feasible


def build_pair_features_and_mask(tasks, servers, current_time):
    """
    输出:
      pair_features: [T, S, 5]
      mask: [T, S] bool
    """
    T = len(tasks)
    S = len(servers)

    pair_features = np.zeros((T, S, 5), dtype=np.float32)
    mask = np.zeros((T, S), dtype=bool)

    for i, task in enumerate(tasks):
        for j, server in enumerate(servers):
            tx_delay, exec_time, total_latency, deadline_margin, energy_cost, feasible = \
                estimate_pair_values(task, server, current_time)

            pair_features[i, j, :] = [
                tx_delay,
                exec_time,
                total_latency,
                deadline_margin,
                energy_cost,
            ]
            mask[i, j] = feasible

    return pair_features, mask


def build_pair_labels(num_tasks, num_servers, teacher_assignment):
    """
    teacher_assignment:
      dict {task_idx: server_idx}
      或 list，assignment[t] = s
    """
    labels = np.zeros((num_tasks, num_servers), dtype=np.float32)

    if isinstance(teacher_assignment, dict):
        for t, s in teacher_assignment.items():
            t = int(t)
            s = int(s)
            if 0 <= t < num_tasks and 0 <= s < num_servers:
                labels[t, s] = 1.0
    else:
        for t, s in enumerate(teacher_assignment):
            if s is None:
                continue
            s = int(s)
            if 0 <= s < num_servers:
                labels[t, s] = 1.0

    return labels


def create_sample(tasks, servers, current_time, teacher_assignment):
    task_features = build_task_features(tasks, current_time)
    server_features = build_server_features(servers)
    pair_features, mask = build_pair_features_and_mask(tasks, servers, current_time)
    labels = build_pair_labels(len(tasks), len(servers), teacher_assignment)

    labels = labels * mask.astype(np.float32)

    return {
        "task_features": task_features,
        "server_features": server_features,
        "pair_features": pair_features,
        "mask": mask,
        "labels": labels,
    }


def save_samples(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# model
# =========================

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
    低开销 GNN:
    - task encoder
    - server encoder
    - pair encoder
    - 一轮轻量 message passing
    - pair scoring
    """

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
        """
        task_feat: [T, Dt]
        server_feat: [S, Ds]
        pair_feat: [T, S, Dp]
        mask: [T, S] bool
        """
        T = task_feat.size(0)
        S = server_feat.size(0)

        t_emb = self.task_encoder(task_feat)         # [T, E]
        s_emb = self.server_encoder(server_feat)     # [S, E]
        p_emb = self.pair_encoder(pair_feat)         # [T, S, E]

        valid = mask.float().unsqueeze(-1)

        s_expand = s_emb.unsqueeze(0).expand(T, S, -1)
        t_expand = t_emb.unsqueeze(1).expand(T, S, -1)

        # task <- server aggregation
        task_msgs = (s_expand + p_emb) * valid
        task_den = valid.sum(dim=1).clamp_min(1.0)
        task_aggr = task_msgs.sum(dim=1) / task_den
        t_emb2 = self.task_update(torch.cat([t_emb, task_aggr], dim=-1))

        # server <- task aggregation
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


# =========================
# loss / train
# =========================

def masked_bce_loss(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    loss = loss * mask.float()
    return loss.sum() / (mask.float().sum() + 1e-8)


def to_tensor_sample(sample, device):
    return {
        "task_features": torch.tensor(sample["task_features"], dtype=torch.float32, device=device),
        "server_features": torch.tensor(sample["server_features"], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(sample["pair_features"], dtype=torch.float32, device=device),
        "mask": torch.tensor(sample["mask"], dtype=torch.bool, device=device),
        "labels": torch.tensor(sample["labels"], dtype=torch.float32, device=device),
    }


def evaluate(model, samples, device="cpu"):
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


def train_minimal_gnn(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    save_path: str,
    hidden_dim=64,
    emb_dim=32,
    dropout=0.0,
    lr=1e-3,
    epochs=20,
    seed=42,
    device="cpu",
):
    set_seed(seed)

    model = MinimalBipartiteGNN(
        task_dim=5,
        server_dim=5,
        pair_dim=5,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
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
        val_loss = evaluate(model, val_samples, device=device)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[MinimalGNN] saved best checkpoint -> {save_path}")

    return model
