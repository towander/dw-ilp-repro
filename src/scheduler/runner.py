from src.baselines.greedy import run_greedy
from src.scheduler.fixed_window import run_fixed_window
from src.scheduler.dynamic_window import run_dynamic_window
from src.scheduler.race import run_fixed_window_race, run_dynamic_window_race
from src.scheduler.gnn_scheduler import run_gnn


_MINIMAL_GNN_SCHEDULER = None


def _get_minimal_gnn_scheduler():
    global _MINIMAL_GNN_SCHEDULER

    if _MINIMAL_GNN_SCHEDULER is None:
        from src.learning_baselines.minimal_gnn_scheduler import GNNScheduler

        _MINIMAL_GNN_SCHEDULER = GNNScheduler(
            checkpoint_path="artifacts/minimal_gnn/minimal_gnn.pt",
            task_dim=5,
            server_dim=5,
            pair_dim=5,
            hidden_dim=64,
            emb_dim=32,
            dropout=0.0,
            device="cpu",
            unique_server=False,
        )

    return _MINIMAL_GNN_SCHEDULER


def run_scheduler(name, tasks, nodes, config):
    if name == "greedy":
        return run_greedy(tasks, nodes)

    elif name == "fixed_window":
        return run_fixed_window(
            tasks,
            nodes,
            config,
            use_pruning=False,
            scheduler_name="fixed_window",
        )

    elif name == "fixed_window_pruned":
        return run_fixed_window(
            tasks,
            nodes,
            config,
            use_pruning=True,
            scheduler_name="fixed_window_pruned",
        )

    elif name == "fixed_window_race":
        return run_fixed_window_race(
            tasks,
            nodes,
            config,
        )

    elif name == "dynamic_window":
        return run_dynamic_window(
            tasks,
            nodes,
            config,
            use_pruning=False,
            scheduler_name="dynamic_window",
        )

    elif name == "dynamic_window_pruned":
        return run_dynamic_window(
            tasks,
            nodes,
            config,
            use_pruning=True,
            scheduler_name="dynamic_window_pruned",
        )

    elif name == "dynamic_window_race":
        return run_dynamic_window_race(
            tasks,
            nodes,
            config,
        )

    elif name == "dw_ilp":
        raise NotImplementedError("Scheduler 'dw_ilp' is not implemented yet.")

    elif name == "gnn":
        return run_gnn(tasks, nodes, config)

    elif name == "minimal_gnn":
        scheduler = _get_minimal_gnn_scheduler()

        gnn_tasks = []
        for t in tasks:
            gnn_tasks.append(
                {
                    "task_id": t.get("task_id", t.get("id")),
                    "required_compute": t.get("required_compute", t.get("workload", 0.0)),
                    "input_size": t.get("input_size", t.get("data_size", 0.0)),
                    "deadline": t.get("deadline", 0.0),
                    "arrival_time": t.get("arrival_time", 0.0),
                    "priority": t.get("priority", 1.0),
                }
            )

        gnn_servers = []
        for n in nodes:
            gnn_servers.append(
                {
                    "node_id": n.get("node_id"),
                    "compute_capacity": n.get("compute_capacity", 0.0),
                    "queue_length": n.get("queue_length", 0.0),
                    "remaining_energy": n.get(
                        "remaining_energy",
                        n.get("battery_level", float("inf")),
                    ),
                }
            )

        assignment, sched_info = scheduler.schedule(
            tasks=gnn_tasks,
            servers=gnn_servers,
            current_time=0.0,
            env_state=None,
        )

        num_assigned = sched_info.get("num_assigned", 0)
        total_pairs = len(tasks) * len(nodes)

        return {
            "scheduler": "minimal_gnn",
            "num_tasks": len(tasks),
            "num_assigned": num_assigned,
            "assignment_ratio": num_assigned / max(1, len(tasks)),
            "avg_completion_time": 0.0,
            "sla_violation_rate": 0.0,
            "allocation_time": sched_info.get("allocation_time_ms", 0.0) / 1000.0,
            "execution_time": 0.0,
            "num_dispatch_rounds": 1,
            "avg_window_size": float(len(tasks)),
            "candidate_pairs_before": total_pairs,
            "candidate_pairs_after": num_assigned,
            "candidate_keep_ratio": num_assigned / max(1, total_pairs),
            "candidate_prune_ratio": 1.0 - num_assigned / max(1, total_pairs),
            "energy_pruned_pairs": 0,
            "time_pruned_pairs": 0,
            "queue_pruned_pairs": 0,
            "forced_energy_overdraw_count": 0,
            "avg_energy_wait_time": 0.0,
            "avg_remaining_energy": 0.0,
            "min_remaining_energy": 0.0,
            "total_remaining_energy": 0.0,
            "energy_enabled_nodes": sum(
                1 for n in nodes if n.get("energy_enabled", False)
            ),
            "assignment": assignment,
            "sched_info": sched_info,
        }

    else:
        raise ValueError(f"Unknown scheduler: {name}")
