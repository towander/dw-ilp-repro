# src/scheduler/gnn_scheduler.py

import time
import numpy as np
import torch

from src.learning_baselines.minimal_gnn_train import (
    MinimalBipartiteGNN,
    build_task_features,
    build_server_features,
    build_pair_features_and_mask,
)


class GNNScheduler:
    """
    低开销 GNN baseline 调度器

    在线阶段：
    - 特征构造
    - 一次轻量前向
    - 按分数贪心选可行 server

    设计目标：
    - 开销低于 ILP
    - 高于 Greedy
    """

    def __init__(
        self,
        checkpoint_path="artifacts/minimal_gnn/minimal_gnn.pt",
        task_dim=5,
        server_dim=5,
        pair_dim=5,
        hidden_dim=64,
        emb_dim=32,
        dropout=0.0,
        device="cpu",
        unique_server=True,
    ):
        self.device = device
        self.unique_server = unique_server

        self.model = MinimalBipartiteGNN(
            task_dim=task_dim,
            server_dim=server_dim,
            pair_dim=pair_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            dropout=dropout,
        ).to(device)

        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def schedule(self, tasks, servers, current_time=0.0, env_state=None):
        """
        输入:
          tasks: list[dict or obj]
          servers: list[dict or obj]
          current_time: float

        输出:
          assignment: dict {task_idx: server_idx}
          info: dict
        """
        total_t0 = time.perf_counter()

        if len(tasks) == 0 or len(servers) == 0:
            return {}, {
                "allocation_time_ms": 0.0,
                "feature_time_ms": 0.0,
                "gnn_inference_time_ms": 0.0,
                "postprocess_time_ms": 0.0,
                "num_tasks": len(tasks),
                "num_servers": len(servers),
                "num_assigned": 0,
            }

        task_features = build_task_features(tasks, current_time)
        server_features = build_server_features(servers)
        pair_features, mask = build_pair_features_and_mask(tasks, servers, current_time)

        feature_t1 = time.perf_counter()

        task_features_t = torch.tensor(task_features, dtype=torch.float32, device=self.device)
        server_features_t = torch.tensor(server_features, dtype=torch.float32, device=self.device)
        pair_features_t = torch.tensor(pair_features, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        infer_t0 = time.perf_counter()
        logits = self.model(task_features_t, server_features_t, pair_features_t, mask_t)
        infer_t1 = time.perf_counter()

        scores = logits.cpu().numpy()
        T, S = scores.shape

        assignment = {}
        used_servers = set()

        # 先处理“最有把握”的 task
        task_best = scores.max(axis=1)
        task_order = np.argsort(-task_best)

        post_t0 = time.perf_counter()

        for t in task_order:
            ranked_servers = np.argsort(-scores[t])
            for s in ranked_servers:
                if not mask[t, s]:
                    continue
                if self.unique_server and (s in used_servers):
                    continue

                assignment[int(t)] = int(s)
                if self.unique_server:
                    used_servers.add(int(s))
                break

        post_t1 = time.perf_counter()
        total_t1 = time.perf_counter()

        info = {
            "allocation_time_ms": (total_t1 - total_t0) * 1000.0,
            "feature_time_ms": (feature_t1 - total_t0) * 1000.0,
            "gnn_inference_time_ms": (infer_t1 - infer_t0) * 1000.0,
            "postprocess_time_ms": (post_t1 - post_t0) * 1000.0,
            "num_tasks": len(tasks),
            "num_servers": len(servers),
            "num_assigned": len(assignment),
        }
        return assignment, info

    # 兼容有些框架用 allocate(...)
    def allocate(self, tasks, servers, current_time=0.0, env_state=None):
        return self.schedule(tasks, servers, current_time=current_time, env_state=env_state)

MinimalGNNScheduler = GNNScheduler
