import copy
import heapq
import math


def _read(obj, keys, default=None):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def task_id(task, idx=0):
    return _read(task, ["task_id", "id"], default=idx)


def node_id(node, idx=0):
    return _read(node, ["node_id", "id"], default=idx)


def arrival_time(task):
    return float(_read(task, ["arrival_time", "arrival", "submit_time", "release_time"], default=0.0))


def deadline(task):
    d = _read(task, ["deadline", "deadline_budget", "latency_req", "sla", "max_latency"], default=None)
    if d is None:
        return float("inf")
    return float(d)


def workload(task):
    return float(_read(task, ["required_flops", "workload", "cpu_cycles", "cycles", "compute_demand"], default=1.0))


def data_size(task):
    return float(_read(task, ["data_size", "input_size", "size_mb", "size"], default=0.0))


def compute_capacity(node):
    return float(_read(node, ["compute_capacity", "cpu_freq", "mips", "capacity"], default=1.0))


def bandwidth(node):
    return float(_read(node, ["bandwidth", "uplink_bandwidth", "tx_rate"], default=1.0))


def power_coeff(node):
    return float(_read(node, ["power_coeff", "energy_coeff", "power"], default=1.0))


def build_task_profile(task):
    arr = arrival_time(task)
    ddl = deadline(task)
    wl = workload(task)
    ds = data_size(task)

    if isinstance(task, dict):
        out = dict(task)
        out["__arr"] = arr
        out["__ddl"] = ddl
        out["__wl"] = wl
        out["__ds"] = ds
        return out

    return {
        "task_id": task_id(task),
        "arrival_time": arr,
        "deadline": ddl,
        "__arr": arr,
        "__ddl": ddl,
        "__wl": wl,
        "__ds": ds,
        "__raw_task": task,
    }


def build_node_state(node, idx=0):
    energy_enabled = bool(_read(node, ["energy_enabled"], default=False))
    battery_capacity = float(_read(node, ["battery_capacity"], default=float("inf")))
    battery_level = float(_read(node, ["battery_level"], default=battery_capacity))
    harvest_rate = float(_read(node, ["harvest_rate"], default=0.0))
    min_reserve_energy = float(_read(node, ["min_reserve_energy"], default=0.0))
    tx_energy_coeff = float(_read(node, ["tx_energy_coeff"], default=0.0))

    return {
        "id": node_id(node, idx),
        "obj": node,
        "available_time": 0.0,
        "compute_capacity": compute_capacity(node),
        "bandwidth": bandwidth(node),
        "power_coeff": power_coeff(node),

        # energy-aware state
        "energy_enabled": energy_enabled,
        "battery_capacity": battery_capacity,
        "battery_level": battery_level,
        "harvest_rate": harvest_rate,
        "min_reserve_energy": min_reserve_energy,
        "tx_energy_coeff": tx_energy_coeff,
        "last_energy_update_time": 0.0,
    }


def clone_node_states(node_states):
    return copy.deepcopy(node_states)


def estimate_transfer_time_fast(task, node_state):
    ds = task["__ds"]
    if ds <= 0:
        return 0.0
    return ds / max(node_state["bandwidth"], 1e-9)


def estimate_exec_time_fast(task, node_state):
    return task["__wl"] / max(node_state["compute_capacity"], 1e-9)


def estimate_finish_time_fast(task, node_state, decision_time):
    start_t = max(decision_time, node_state["available_time"], task["__arr"])
    return start_t + estimate_transfer_time_fast(task, node_state) + estimate_exec_time_fast(task, node_state)


def estimate_energy_fast(task, node_state):
    exec_energy = node_state["power_coeff"] * estimate_exec_time_fast(task, node_state)
    tx_energy = node_state.get("tx_energy_coeff", 0.0) * task["__ds"]
    return exec_energy + tx_energy


def _project_battery_level(node_state, now_time):
    if not node_state.get("energy_enabled", False):
        return float("inf")

    last_t = float(node_state.get("last_energy_update_time", 0.0))
    battery_level = float(node_state.get("battery_level", 0.0))
    battery_capacity = float(node_state.get("battery_capacity", battery_level))
    harvest_rate = float(node_state.get("harvest_rate", 0.0))

    if now_time <= last_t:
        return battery_level

    dt = now_time - last_t
    return min(battery_capacity, battery_level + harvest_rate * dt)


def _sync_energy_state(node_state, now_time):
    if not node_state.get("energy_enabled", False):
        return

    node_state["battery_level"] = _project_battery_level(node_state, now_time)
    node_state["last_energy_update_time"] = now_time


def _energy_wait_time(task, node_state, base_start, safety_factor=1.0):
    if not node_state.get("energy_enabled", False):
        return 0.0

    projected_battery = _project_battery_level(node_state, base_start)
    required_energy = safety_factor * estimate_energy_fast(task, node_state)
    reserve = float(node_state.get("min_reserve_energy", 0.0))

    deficit = reserve + required_energy - projected_battery
    if deficit <= 0:
        return 0.0

    harvest_rate = float(node_state.get("harvest_rate", 0.0))
    if harvest_rate <= 1e-12:
        return float("inf")

    return deficit / harvest_rate


def _is_energy_prunable(task, node_state, decision_time, pruning_cfg):
    """
    Return True if the node is clearly unattractive or infeasible under energy-aware rule.
    """
    if not node_state.get("energy_enabled", False):
        return False

    energy_safety_factor = float(pruning_cfg.get("energy_safety_factor", 1.05))
    max_extra_wait = float(pruning_cfg.get("energy_wait_threshold", 0.20))

    transfer_t = estimate_transfer_time_fast(task, node_state)
    exec_t = estimate_exec_time_fast(task, node_state)
    service_t = transfer_t + exec_t

    base_start = max(decision_time, node_state["available_time"], task["__arr"])
    extra_wait = _energy_wait_time(task, node_state, base_start, safety_factor=energy_safety_factor)

    if math.isinf(extra_wait):
        return True

    projected_finish = base_start + extra_wait + service_t
    projected_completion = projected_finish - task["__arr"]

    if extra_wait > max_extra_wait:
        return True

    if projected_completion > task["__ddl"]:
        return True

    return False


def get_candidate_node_states(task, node_states, pruning_cfg=None, decision_time=None, use_pruning=False):
    before = len(node_states)

    if (not use_pruning) or (pruning_cfg is None):
        return list(node_states), {
            "before": before,
            "after": before,
            "energy_pruned": 0,
            "time_pruned": 0,
            "queue_pruned": 0,
        }

    prune_time_factor = float(pruning_cfg.get("prune_time_factor", 0.60))
    queue_wait_threshold = float(pruning_cfg.get("queue_wait_threshold", 0.12))
    enable_topk = bool(pruning_cfg.get("enable_topk", True))
    top_k_candidates = int(pruning_cfg.get("top_k_candidates", 4))
    enable_energy_pruning = bool(pruning_cfg.get("enable_energy_pruning", True))

    if decision_time is None:
        decision_time = task["__arr"]

    effective_now = max(decision_time, task["__arr"])
    service_threshold = prune_time_factor * task["__ddl"]

    filtered = []
    time_pruned = 0
    queue_pruned = 0
    energy_pruned = 0

    for ns in node_states:
        transfer_t = estimate_transfer_time_fast(task, ns)
        exec_t = estimate_exec_time_fast(task, ns)
        service_t = transfer_t + exec_t

        if service_t > service_threshold:
            time_pruned += 1
            continue

        est_queue_wait = max(0.0, ns["available_time"] - effective_now)
        if est_queue_wait > queue_wait_threshold:
            queue_pruned += 1
            continue

        if enable_energy_pruning and _is_energy_prunable(task, ns, decision_time, pruning_cfg):
            energy_pruned += 1
            continue

        filtered.append(ns)

    if not filtered:
        # 回退，避免剪枝过激导致空候选集
        filtered = list(node_states)

    if enable_topk and top_k_candidates > 0 and len(filtered) > top_k_candidates:
        filtered = heapq.nsmallest(
            top_k_candidates,
            filtered,
            key=lambda ns: estimate_finish_time_fast(task, ns, decision_time),
        )

    after = len(filtered)
    return filtered, {
        "before": before,
        "after": after,
        "energy_pruned": energy_pruned,
        "time_pruned": time_pruned,
        "queue_pruned": queue_pruned,
    }


def choose_best_node_for_task(task, node_states, decision_time, use_pruning=False, pruning_cfg=None):
    if pruning_cfg is None:
        pruning_cfg = {}

    energy_safety_factor = float(pruning_cfg.get("energy_safety_factor", 1.00))

    candidate_states, prune_stats = get_candidate_node_states(
        task=task,
        node_states=node_states,
        pruning_cfg=pruning_cfg,
        decision_time=decision_time,
        use_pruning=use_pruning,
    )

    def _select_best(states, allow_forced_violation=False):
        best_ns = None
        best_start = None
        best_finish = None
        best_transfer = None
        best_exec = None
        best_energy = None
        best_energy_wait = None
        best_battery_before = None
        best_battery_after = None
        best_forced = False

        for ns in states:
            transfer_t = estimate_transfer_time_fast(task, ns)
            exec_t = estimate_exec_time_fast(task, ns)
            energy = estimate_energy_fast(task, ns)

            base_start = max(decision_time, ns["available_time"], task["__arr"])
            extra_wait = _energy_wait_time(
                task,
                ns,
                base_start,
                safety_factor=energy_safety_factor,
            )

            if math.isinf(extra_wait):
                if not allow_forced_violation:
                    continue
                extra_wait = 0.0
                forced = True
            else:
                forced = False

            start_t = base_start + extra_wait
            battery_before = _project_battery_level(ns, start_t)
            battery_after = battery_before - energy

            if ns.get("energy_enabled", False):
                reserve = float(ns.get("min_reserve_energy", 0.0))
                feasible = (battery_after >= reserve - 1e-9)
                if (not feasible) and (not allow_forced_violation):
                    continue
                if not feasible:
                    forced = True

            finish_t = start_t + transfer_t + exec_t

            if (
                best_finish is None
                or finish_t < best_finish
                or (finish_t == best_finish and energy < best_energy)
            ):
                best_ns = ns
                best_start = start_t
                best_finish = finish_t
                best_transfer = transfer_t
                best_exec = exec_t
                best_energy = energy
                best_energy_wait = max(0.0, extra_wait)
                best_battery_before = battery_before
                best_battery_after = battery_after
                best_forced = forced

        return {
            "node_state": best_ns,
            "start_time": best_start,
            "finish_time": best_finish,
            "transfer_time": best_transfer,
            "execution_time": best_exec,
            "energy": best_energy,
            "energy_wait_time": best_energy_wait,
            "battery_before": best_battery_before,
            "battery_after": best_battery_after,
            "forced_energy_violation": best_forced,
        }

    chosen = _select_best(candidate_states, allow_forced_violation=False)

    # 剪枝后没找到，再在全体节点里找
    if chosen["node_state"] is None:
        chosen = _select_best(node_states, allow_forced_violation=False)

    # 如果依然找不到，最后兜底：允许强制落点，避免整个实验中断
    if chosen["node_state"] is None:
        chosen = _select_best(node_states, allow_forced_violation=True)

    chosen["candidate_pairs_before"] = prune_stats["before"]
    chosen["candidate_pairs_after"] = prune_stats["after"]
    chosen["energy_pruned_pairs"] = prune_stats["energy_pruned"]
    chosen["time_pruned_pairs"] = prune_stats["time_pruned"]
    chosen["queue_pruned_pairs"] = prune_stats["queue_pruned"]
    return chosen


def dispatch_batch(batch, node_states, decision_time, sort_key_fn, use_pruning=False, pruning_cfg=None):
    batch_sorted = sorted(batch, key=sort_key_fn)

    assignments = []
    completion_times = []
    num_violations = 0
    total_exec_time = 0.0
    candidate_pairs_before = 0
    candidate_pairs_after = 0
    energy_pruned_pairs = 0
    time_pruned_pairs = 0
    queue_pruned_pairs = 0
    forced_energy_overdraw_count = 0
    total_energy_wait_time = 0.0

    batch_start = min(t["__arr"] for t in batch)
    batch_window_size = decision_time - batch_start

    for idx, task in enumerate(batch_sorted):
        chosen = choose_best_node_for_task(
            task=task,
            node_states=node_states,
            decision_time=decision_time,
            use_pruning=use_pruning,
            pruning_cfg=pruning_cfg,
        )

        candidate_pairs_before += chosen["candidate_pairs_before"]
        candidate_pairs_after += chosen["candidate_pairs_after"]
        energy_pruned_pairs += chosen["energy_pruned_pairs"]
        time_pruned_pairs += chosen["time_pruned_pairs"]
        queue_pruned_pairs += chosen["queue_pruned_pairs"]
        total_energy_wait_time += chosen.get("energy_wait_time", 0.0) or 0.0

        ns = chosen["node_state"]

        if ns.get("energy_enabled", False):
            _sync_energy_state(ns, chosen["start_time"])
            ns["battery_level"] = max(0.0, ns["battery_level"] - chosen["energy"])
            ns["last_energy_update_time"] = chosen["start_time"]

        if chosen.get("forced_energy_violation", False):
            forced_energy_overdraw_count += 1

        ns["available_time"] = chosen["finish_time"]

        comp_time = chosen["finish_time"] - task["__arr"]
        violated = comp_time > task["__ddl"]
        if violated:
            num_violations += 1

        completion_times.append(comp_time)
        total_exec_time += chosen["execution_time"]

        assignments.append({
            "task_id": task_id(task, idx),
            "node_id": ns["id"],
            "arrival_time": task["__arr"],
            "window_start": batch_start,
            "window_end": decision_time,
            "window_size": batch_window_size,
            "start_time": chosen["start_time"],
            "finish_time": chosen["finish_time"],
            "completion_time": comp_time,
            "deadline": task["__ddl"],
            "sla_violated": violated,
            "transfer_time": chosen["transfer_time"],
            "execution_time": chosen["execution_time"],
            "energy": chosen["energy"],
            "energy_wait_time": chosen.get("energy_wait_time", 0.0),
            "battery_before": chosen.get("battery_before"),
            "battery_after": ns.get("battery_level"),
            "forced_energy_violation": chosen.get("forced_energy_violation", False),
        })

    return {
        "assignments": assignments,
        "completion_times": completion_times,
        "num_violations": num_violations,
        "total_exec_time": total_exec_time,
        "candidate_pairs_before": candidate_pairs_before,
        "candidate_pairs_after": candidate_pairs_after,
        "energy_pruned_pairs": energy_pruned_pairs,
        "time_pruned_pairs": time_pruned_pairs,
        "queue_pruned_pairs": queue_pruned_pairs,
        "forced_energy_overdraw_count": forced_energy_overdraw_count,
        "total_energy_wait_time": total_energy_wait_time,
        "window_size": batch_window_size,
    }


def summarize_energy_states(node_states):
    energy_nodes = [ns for ns in node_states if ns.get("energy_enabled", False)]
    if not energy_nodes:
        return {
            "avg_remaining_energy": 0.0,
            "min_remaining_energy": 0.0,
            "total_remaining_energy": 0.0,
            "energy_enabled_nodes": 0,
        }

    levels = []
    for ns in energy_nodes:
        lvl = _project_battery_level(ns, ns.get("available_time", 0.0))
        levels.append(lvl)

    return {
        "avg_remaining_energy": sum(levels) / len(levels),
        "min_remaining_energy": min(levels),
        "total_remaining_energy": sum(levels),
        "energy_enabled_nodes": len(levels),
    }
