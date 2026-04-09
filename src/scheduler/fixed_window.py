import time

from src.scheduler.pruning import (
    arrival_time,
    build_node_state,
    build_task_profile,
    dispatch_batch,
    summarize_energy_states,
)


def _task_sort_key(task, sort_by):
    if sort_by == "arrival":
        return (task["__arr"], task["__ddl"])
    if sort_by == "workload":
        return (task["__wl"], task["__ddl"])
    return (task["__ddl"], task["__arr"])


def plan_fixed_window(tasks, nodes, config, use_pruning=False, scheduler_name="fixed_window"):
    start_alloc_clock = time.perf_counter()

    fw_cfg = config.get("fixed_window", {})
    pruning_cfg = config.get("pruning", {})

    window_size = float(fw_cfg.get("window_size", 0.30))
    sort_by = fw_cfg.get("sort_by", "deadline")

    if len(tasks) == 0:
        return {
            "scheduler": scheduler_name,
            "num_tasks": 0,
            "avg_completion_time": 0.0,
            "sla_violation_rate": 0.0,
            "allocation_time": 0.0,
            "execution_time": 0.0,
            "num_dispatch_rounds": 0,
            "avg_window_size": 0.0,
            "candidate_pairs_before": 0,
            "candidate_pairs_after": 0,
            "candidate_keep_ratio": 1.0,
            "candidate_prune_ratio": 0.0,
            "energy_pruned_pairs": 0,
            "time_pruned_pairs": 0,
            "queue_pruned_pairs": 0,
            "forced_energy_overdraw_count": 0,
            "avg_energy_wait_time": 0.0,
            "avg_remaining_energy": 0.0,
            "min_remaining_energy": 0.0,
            "total_remaining_energy": 0.0,
            "assignments": [],
        }

    tasks_sorted = [build_task_profile(t) for t in sorted(tasks, key=lambda t: arrival_time(t))]
    node_states = [build_node_state(node, i) for i, node in enumerate(nodes)]

    all_assignments = []
    all_completion_times = []
    total_violations = 0
    total_exec_time = 0.0
    candidate_pairs_before = 0
    candidate_pairs_after = 0
    energy_pruned_pairs = 0
    time_pruned_pairs = 0
    queue_pruned_pairs = 0
    forced_energy_overdraw_count = 0
    total_energy_wait_time = 0.0
    num_dispatch_rounds = 0
    total_window_size = 0.0

    n = len(tasks_sorted)
    ptr = 0

    while ptr < n:
        batch_start = tasks_sorted[ptr]["__arr"]
        decision_time = batch_start + window_size

        batch = []
        while ptr < n and tasks_sorted[ptr]["__arr"] <= decision_time:
            batch.append(tasks_sorted[ptr])
            ptr += 1

        result = dispatch_batch(
            batch=batch,
            node_states=node_states,
            decision_time=decision_time,
            sort_key_fn=lambda task: _task_sort_key(task, sort_by),
            use_pruning=use_pruning,
            pruning_cfg=pruning_cfg,
        )

        all_assignments.extend(result["assignments"])
        all_completion_times.extend(result["completion_times"])
        total_violations += result["num_violations"]
        total_exec_time += result["total_exec_time"]
        candidate_pairs_before += result["candidate_pairs_before"]
        candidate_pairs_after += result["candidate_pairs_after"]
        energy_pruned_pairs += result["energy_pruned_pairs"]
        time_pruned_pairs += result["time_pruned_pairs"]
        queue_pruned_pairs += result["queue_pruned_pairs"]
        forced_energy_overdraw_count += result["forced_energy_overdraw_count"]
        total_energy_wait_time += result["total_energy_wait_time"]
        num_dispatch_rounds += 1
        total_window_size += result["window_size"]

    allocation_time = time.perf_counter() - start_alloc_clock
    num_tasks = len(tasks_sorted)
    avg_completion_time = sum(all_completion_times) / num_tasks if num_tasks > 0 else 0.0
    sla_violation_rate = total_violations / num_tasks if num_tasks > 0 else 0.0
    avg_window_size = total_window_size / num_dispatch_rounds if num_dispatch_rounds > 0 else 0.0
    candidate_keep_ratio = (candidate_pairs_after / candidate_pairs_before) if candidate_pairs_before > 0 else 1.0
    candidate_prune_ratio = 1.0 - candidate_keep_ratio
    avg_energy_wait_time = total_energy_wait_time / num_tasks if num_tasks > 0 else 0.0

    energy_summary = summarize_energy_states(node_states)

    return {
        "scheduler": scheduler_name,
        "num_tasks": num_tasks,
        "avg_completion_time": avg_completion_time,
        "sla_violation_rate": sla_violation_rate,
        "allocation_time": allocation_time,
        "execution_time": total_exec_time,
        "num_dispatch_rounds": num_dispatch_rounds,
        "avg_window_size": avg_window_size,
        "candidate_pairs_before": candidate_pairs_before,
        "candidate_pairs_after": candidate_pairs_after,
        "candidate_keep_ratio": candidate_keep_ratio,
        "candidate_prune_ratio": candidate_prune_ratio,
        "energy_pruned_pairs": energy_pruned_pairs,
        "time_pruned_pairs": time_pruned_pairs,
        "queue_pruned_pairs": queue_pruned_pairs,
        "forced_energy_overdraw_count": forced_energy_overdraw_count,
        "avg_energy_wait_time": avg_energy_wait_time,
        **energy_summary,
        "assignments": all_assignments,
    }


def run_fixed_window(tasks, nodes, config, use_pruning=False, scheduler_name="fixed_window"):
    return plan_fixed_window(
        tasks=tasks,
        nodes=nodes,
        config=config,
        use_pruning=use_pruning,
        scheduler_name=scheduler_name,
    )
