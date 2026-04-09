import random
import math


def _uniform(cfg, key_min, key_max, default_min, default_max):
    return random.uniform(
        cfg.get(key_min, default_min),
        cfg.get(key_max, default_max)
    )


def _sample_arrival_time(sim_cfg):
    """
    Generate arrival times using a Poisson-process style inter-arrival time:
    delta_t ~ Exponential(arrival_rate)
    """
    total_time = float(sim_cfg.get("total_time", 100))
    arrival_rate = float(sim_cfg.get("arrival_rate", 1.0))

    if arrival_rate <= 0:
        return random.uniform(0, total_time)

    t = 0.0
    arrivals = []
    while t < total_time:
        delta = random.expovariate(arrival_rate)
        t += delta
        if t < total_time:
            arrivals.append(t)
    return arrivals


def _estimate_min_edge_time(task, nodes):
    """
    Estimate the minimum possible (transfer + execution) time
    among all candidate edge nodes.
    """
    best = float("inf")

    workload = task["workload"]
    data_size = task["data_size"]

    for node in nodes:
        capacity = max(node.get("compute_capacity", 1.0), 1e-9)
        bandwidth = max(node.get("bandwidth", 1.0), 1e-9)

        exec_t = workload / capacity
        tx_t = data_size / bandwidth
        total = tx_t + exec_t

        if total < best:
            best = total

    return best


def _is_edge_offloadable(task, nodes, admission_cfg):
    """
    Filtering rule for edge-scheduling task generation.

    Logic:
    1. Extremely small deadlines are treated as local-only tasks.
    2. For edge tasks, there must exist at least one node such that:
         min(tx + exec) <= feasibility_margin * deadline
    """
    deadline = task["deadline"]

    local_threshold = admission_cfg.get("local_deadline_threshold", 0.3)
    if deadline < local_threshold:
        return False

    min_edge_time = _estimate_min_edge_time(task, nodes)
    margin = admission_cfg.get("feasibility_margin", 0.7)

    return min_edge_time <= margin * deadline


def _generate_one_task(task_id, arrival_time, task_cfg):
    task = {
        "id": task_id,
        "arrival_time": arrival_time,
        "workload": _uniform(task_cfg, "workload_min", "workload_max", 20.0, 80.0),
        "data_size": _uniform(task_cfg, "data_size_min", "data_size_max", 1.0, 10.0),
        "deadline": _uniform(task_cfg, "deadline_min", "deadline_max", 1.0, 3.0),
    }
    return task


def generate_tasks(config, nodes=None):
    """
    Generate offload-eligible Edge-AI tasks.

    Arrival model:
    - Poisson-like arrivals using exponential inter-arrival time based on arrival_rate.
    - If generated arrival count is not enough, we continue sampling until num_tasks accepted.
    """
    sim_cfg = config.get("simulation", {})
    task_cfg = config.get("task", {})
    admission_cfg = config.get("admission", {})

    num_tasks = int(sim_cfg.get("num_tasks", 20))
    total_time = float(sim_cfg.get("total_time", 100))
    arrival_rate = float(sim_cfg.get("arrival_rate", 1.0))

    edge_only = admission_cfg.get("edge_only", True)
    max_trials = admission_cfg.get("max_regen_trials", 200)

    tasks = []
    task_id = 0
    total_trials = 0

    current_time = 0.0

    while len(tasks) < num_tasks:
        if total_trials >= max_trials * num_tasks:
            raise RuntimeError(
                "Task generation failed: too many rejected samples. "
                "Please relax task deadlines or admission constraints."
            )

        if arrival_rate > 0:
            current_time += random.expovariate(arrival_rate)
            if current_time >= total_time:
                # 超出仿真时域，重新从头采一个合法时间点，避免任务数不够
                current_time = random.uniform(0, total_time)
        else:
            current_time = random.uniform(0, total_time)

        candidate = _generate_one_task(task_id, current_time, task_cfg)
        total_trials += 1

        if nodes is not None and edge_only:
            accepted = _is_edge_offloadable(candidate, nodes, admission_cfg)
            if not accepted:
                continue

        tasks.append(candidate)
        task_id += 1

    tasks.sort(key=lambda x: x["arrival_time"])
    return tasks
