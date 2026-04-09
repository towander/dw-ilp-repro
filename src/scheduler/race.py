from src.scheduler.fixed_window import plan_fixed_window
from src.scheduler.dynamic_window import plan_dynamic_window


def _merge_race_result(base_res, pruned_res, scheduler_name):
    base_t = float(base_res["allocation_time"])
    pruned_t = float(pruned_res["allocation_time"])

    if pruned_t < base_t:
        winner = "pruned"
        chosen = pruned_res
    else:
        winner = "base"
        chosen = base_res

    race_finish_time = min(base_t, pruned_t)

    final = dict(chosen)
    final["scheduler"] = scheduler_name
    final["allocation_time"] = race_finish_time

    final["base_branch_time"] = base_t
    final["pruned_branch_time"] = pruned_t
    final["race_finish_time"] = race_finish_time
    final["race_winner"] = winner
    final["base_wins"] = 1 if winner == "base" else 0
    final["pruned_wins"] = 1 if winner == "pruned" else 0

    final["base_avg_completion_time"] = base_res["avg_completion_time"]
    final["pruned_avg_completion_time"] = pruned_res["avg_completion_time"]
    final["base_sla_violation_rate"] = base_res["sla_violation_rate"]
    final["pruned_sla_violation_rate"] = pruned_res["sla_violation_rate"]
    final["base_execution_time"] = base_res["execution_time"]
    final["pruned_execution_time"] = pruned_res["execution_time"]

    final["base_candidate_pairs_before"] = base_res.get("candidate_pairs_before", 0)
    final["pruned_candidate_pairs_before"] = pruned_res.get("candidate_pairs_before", 0)
    final["base_candidate_pairs_after"] = base_res.get("candidate_pairs_after", 0)
    final["pruned_candidate_pairs_after"] = pruned_res.get("candidate_pairs_after", 0)

    final["base_energy_pruned_pairs"] = base_res.get("energy_pruned_pairs", 0)
    final["pruned_energy_pruned_pairs"] = pruned_res.get("energy_pruned_pairs", 0)
    final["base_avg_energy_wait_time"] = base_res.get("avg_energy_wait_time", 0.0)
    final["pruned_avg_energy_wait_time"] = pruned_res.get("avg_energy_wait_time", 0.0)
    final["base_avg_remaining_energy"] = base_res.get("avg_remaining_energy", 0.0)
    final["pruned_avg_remaining_energy"] = pruned_res.get("avg_remaining_energy", 0.0)

    final["quality_gap_avg_completion_pruned_minus_base"] = (
        pruned_res["avg_completion_time"] - base_res["avg_completion_time"]
    )
    final["quality_gap_sla_pruned_minus_base"] = (
        pruned_res["sla_violation_rate"] - base_res["sla_violation_rate"]
    )
    final["quality_gap_exec_time_pruned_minus_base"] = (
        pruned_res["execution_time"] - base_res["execution_time"]
    )
    final["quality_gap_energy_wait_pruned_minus_base"] = (
        pruned_res.get("avg_energy_wait_time", 0.0) - base_res.get("avg_energy_wait_time", 0.0)
    )

    return final


def run_fixed_window_race(tasks, nodes, config):
    base_res = plan_fixed_window(
        tasks=tasks,
        nodes=nodes,
        config=config,
        use_pruning=False,
        scheduler_name="fixed_window_base_branch",
    )
    pruned_res = plan_fixed_window(
        tasks=tasks,
        nodes=nodes,
        config=config,
        use_pruning=True,
        scheduler_name="fixed_window_pruned_branch",
    )
    return _merge_race_result(base_res, pruned_res, "fixed_window_race")


def run_dynamic_window_race(tasks, nodes, config):
    base_res = plan_dynamic_window(
        tasks=tasks,
        nodes=nodes,
        config=config,
        use_pruning=False,
        scheduler_name="dynamic_window_base_branch",
    )
    pruned_res = plan_dynamic_window(
        tasks=tasks,
        nodes=nodes,
        config=config,
        use_pruning=True,
        scheduler_name="dynamic_window_pruned_branch",
    )
    return _merge_race_result(base_res, pruned_res, "dynamic_window_race")
