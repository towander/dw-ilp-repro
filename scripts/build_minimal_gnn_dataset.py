# scripts/build_minimal_gnn_dataset.py

import os
import random
import pickle
import numpy as np
import yaml

from src.simulator.task_generator import generate_tasks
from src.simulator.node_generator import generate_nodes
from src.scheduler.dynamic_window import run_dynamic_window


def safe_get(obj, name, default=0.0):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def build_task_features(tasks, current_time=0.0):
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


def estimate_pair_values(task, server, current_time=0.0):
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


def build_pair_features_and_mask(tasks, servers, current_time=0.0):
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
    labels = np.zeros((num_tasks, num_servers), dtype=np.float32)

    # 情况1：已经是 dict {task_idx: server_idx}
    if isinstance(teacher_assignment, dict):
        for t, s in teacher_assignment.items():
            t = int(t)
            s = int(s)
            if 0 <= t < num_tasks and 0 <= s < num_servers:
                labels[t, s] = 1.0
        return labels

    # 情况2：是 assignments 列表
    if isinstance(teacher_assignment, list):
        for item in teacher_assignment:
            if not isinstance(item, dict):
                continue

            # 兼容常见字段名
            t = item.get("task_id", item.get("task_idx", item.get("task")))
            s = item.get("assigned_node", item.get("node_id", item.get("server_idx", item.get("server"))))

            if t is None or s is None:
                continue

            t = int(t)
            s = int(s)

            if 0 <= t < num_tasks and 0 <= s < num_servers:
                labels[t, s] = 1.0

        return labels

    raise ValueError(
        "teacher_assignment must be either dict {task_idx: server_idx} "
        "or a list of assignment dicts"
    )


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


def main():
    with open("config/exp_dynamic_race_large_v1.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 训练集 / 验证集种子
    train_seeds = [101, 102, 103, 104, 105]
    val_seeds = [201, 202]

    # 可调：每个 seed 生成多少个独立实例
    instances_per_seed = 10

    train_samples = []
    val_samples = []

    print("[GNN Dataset] building train set...")
    for seed in train_seeds:
        for rep in range(instances_per_seed):
            data_seed = seed * 1000 + rep
            set_seed(data_seed)

            nodes = generate_nodes(base_config, seed=data_seed)
            tasks = generate_tasks(base_config, nodes)

            result = run_dynamic_window(
                tasks,
                nodes,
                base_config,
                use_pruning=True,
                scheduler_name="dynamic_window_pruned_teacher",
            )
            # print("[DEBUG] result keys =", result.keys())
            # print("[DEBUG] first assignment =", result["assignments"][0] if len(result["assignments"]) > 0 else None)

            if "assignments" not in result:
                raise KeyError(
                "Teacher result does not contain 'assignments'. "
                "Please check src/scheduler/dynamic_window.py."
            )

            sample = create_sample(
                tasks=tasks,
                servers=nodes,
                current_time=0.0,
                teacher_assignment=result["assignments"],
            )
            train_samples.append(sample)

            print(f"[train] seed={seed} rep={rep} samples={len(train_samples)}")

    print("[GNN Dataset] building val set...")
    for seed in val_seeds:
        for rep in range(instances_per_seed):
            data_seed = seed * 1000 + rep
            set_seed(data_seed)

            nodes = generate_nodes(base_config, seed=data_seed)
            tasks = generate_tasks(base_config, nodes)

            result = run_dynamic_window(
                tasks,
                nodes,
                base_config,
                use_pruning=True,
                scheduler_name="dynamic_window_pruned_teacher",
            )

            if "assignments" not in result:
                raise KeyError(
                    "Teacher result does not contain 'assignment'. "
                    "Please modify src/scheduler/dynamic_window.py to return assignment."
                )

            sample = create_sample(
                tasks=tasks,
                servers=nodes,
                current_time=0.0,
                teacher_assignment=result["assignments"],
            )
            val_samples.append(sample)

            print(f"[val] seed={seed} rep={rep} samples={len(val_samples)}")

    save_samples(train_samples, "artifacts/minimal_gnn/train.pkl")
    save_samples(val_samples, "artifacts/minimal_gnn/val.pkl")

    print(f"[GNN Dataset] saved train={len(train_samples)} -> artifacts/minimal_gnn/train.pkl")
    print(f"[GNN Dataset] saved val={len(val_samples)} -> artifacts/minimal_gnn/val.pkl")


if __name__ == "__main__":
    main()
