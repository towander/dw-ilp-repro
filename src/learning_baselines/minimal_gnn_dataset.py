# src/learning_baselines/minimal_gnn_dataset.py

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def safe_get(obj, name, default=0.0):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_task_features(tasks, current_time):
    """
    输出 shape: [T, 5]
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
        deadline = float(safe_get(t, "deadline", current_time))
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
    return np.asarray(feats, dtype=np.float32)


def build_server_features(servers):
    """
    输出 shape: [S, 5]
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
    return np.asarray(feats, dtype=np.float32)


def estimate_pair_values(task, server, current_time):
    """
    你可以先用这个通用估计版。
    如果你项目里已有精确的 tx_delay / exec_time / energy model，
    直接替换这里就行。
    """
    required_compute = float(safe_get(task, "required_compute", safe_get(task, "flops", 1.0)))
    input_size = float(safe_get(task, "input_size", safe_get(task, "data_size", 1.0)))
    deadline = float(safe_get(task, "deadline", current_time + 100.0))

    compute_capacity = float(safe_get(server, "compute_capacity", safe_get(server, "cpu_capacity", 1.0)))
    queue_length = float(safe_get(server, "queue_length", 0.0))
    remaining_energy = float(safe_get(server, "remaining_energy", safe_get(server, "energy", 100.0)))

    # 简单估计
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
      mask:          [T, S] bool
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
      - dict {task_idx: server_idx}
      - 或 list/tuple, assignment[t] = s / -1
    """
    labels = np.zeros((num_tasks, num_servers), dtype=np.float32)

    if isinstance(teacher_assignment, dict):
        for t, s in teacher_assignment.items():
            if 0 <= t < num_tasks and 0 <= s < num_servers:
                labels[t, s] = 1.0
    else:
        for t, s in enumerate(teacher_assignment):
            if 0 <= s < num_servers:
                labels[t, s] = 1.0

    return labels


def create_sample(tasks, servers, current_time, teacher_assignment):
    task_features = build_task_features(tasks, current_time)
    server_features = build_server_features(servers)
    pair_features, mask = build_pair_features_and_mask(tasks, servers, current_time)
    labels = build_pair_labels(len(tasks), len(servers), teacher_assignment)

    # 避免 teacher 落在不可行 pair 上
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


def masked_bce_loss(logits, labels, mask):
    """
    logits: [T, S]
    labels: [T, S]
    mask:   [T, S]
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    loss = loss * mask.float()
    return loss.sum() / (mask.float().sum() + 1e-8)
