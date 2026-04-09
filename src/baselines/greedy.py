import time


def _task_id(task, idx):
    return task.get("task_id", task.get("id", idx))


def _arrival_time(task):
    return float(task.get("arrival_time", 0.0))


def _deadline(task):
    return float(task.get("deadline", task.get("latency_req", float("inf"))))


def _workload(task):
    value = task.get("required_flops", task.get("workload", None))
    if value is None:
        raise KeyError(f"Task {_task_id(task, '?')} missing workload/required_flops")
    return float(value)


def _data_size(task):
    return float(task.get("data_size", task.get("input_size", 0.0)))


def _node_id(node, idx):
    return node.get("node_id", node.get("id", idx))


def _capacity(node):
    value = node.get("compute_capacity", node.get("capacity", None))
    if value is None:
        raise KeyError(f"Node {_node_id(node, '?')} missing compute_capacity/capacity")
    return float(value)


def _bandwidth(node):
    return float(node.get("bandwidth", 1.0))


def _estimate_exec_time(task, node):
    return _workload(task) / max(_capacity(node), 1e-9)


def _estimate_transfer_time(task, node):
    return _data_size(task) / max(_bandwidth(node), 1e-9)


def run_greedy(tasks, nodes):
    """
    Greedy scheduler:
    assign each task to the node that yields the earliest finish time.
    finish_time includes transfer + execution for fair comparison with fixed_window.
    """
    start_alloc_clock = time.perf_counter()

    if len(tasks) == 0:
        return {
            "scheduler": "greedy",
            "num_tasks": 0,
            "avg_completion_time": 0.0,
            "sla_violation_rate": 0.0,
            "allocation_time": 0.0,
            "execution_time": 0.0,
            "task_records": [],
        }

    tasks_sorted = sorted(tasks, key=lambda t: _arrival_time(t))

    node_available_time = {}
    for i, node in enumerate(nodes):
        nid = _node_id(node, i)
        node_available_time[nid] = 0.0

    task_records = []
    total_execution_time = 0.0
    num_violations = 0

    for idx, task in enumerate(tasks_sorted):
        tid = _task_id(task, idx)
        arr_t = _arrival_time(task)
        deadline = _deadline(task)

        best_node_id = None
        best_finish_time = None
        best_start_time = None
        best_exec_time = None
        best_transfer_time = None

        for i, node in enumerate(nodes):
            node_id = _node_id(node, i)

            start_time = max(arr_t, node_available_time[node_id])
            transfer_time = _estimate_transfer_time(task, node)
            exec_time = _estimate_exec_time(task, node)
            finish_time = start_time + transfer_time + exec_time

            if best_finish_time is None or finish_time < best_finish_time:
                best_finish_time = finish_time
                best_start_time = start_time
                best_exec_time = exec_time
                best_transfer_time = transfer_time
                best_node_id = node_id

        node_available_time[best_node_id] = best_finish_time

        completion_time = best_finish_time - arr_t
        violated = completion_time > deadline
        if violated:
            num_violations += 1

        task_records.append({
            "task_id": tid,
            "assigned_node": best_node_id,
            "arrival_time": arr_t,
            "start_time": best_start_time,
            "finish_time": best_finish_time,
            "transfer_time": best_transfer_time,
            "execution_time": best_exec_time,
            "completion_time": completion_time,
            "deadline": deadline,
            "sla_violated": violated,
        })

        total_execution_time += best_exec_time

    num_tasks = len(tasks_sorted)
    avg_completion_time = sum(r["completion_time"] for r in task_records) / num_tasks
    sla_violation_rate = num_violations / num_tasks
    allocation_time = time.perf_counter() - start_alloc_clock

    return {
        "scheduler": "greedy",
        "num_tasks": num_tasks,
        "avg_completion_time": avg_completion_time,
        "sla_violation_rate": sla_violation_rate,
        "allocation_time": allocation_time,
        "execution_time": total_execution_time,
        "task_records": task_records,
    }
