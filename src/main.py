import os
import json
import random
import numpy as np

from src.cli import parse_args
from src.utils.config import load_config
from src.utils.io import save_json, save_csv
from src.simulator.task_generator import generate_tasks
from src.simulator.node_generator import generate_nodes
from src.scheduler.runner import run_scheduler


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _cfg(config, *keys, default=None):
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_config_summary(config, config_path):
    lines = []
    lines.append("=== DW-ILP Reproducibility Entrypoint ===")
    lines.append(f"Loaded config from: {config_path}")
    lines.append("")
    lines.append("=== Effective Config Summary ===")
    lines.append(f"experiment_name: {_cfg(config, 'experiment_name', default='N/A')}")
    lines.append(f"seeds: {config.get('seeds', [])}")
    lines.append(f"num_tasks: {_cfg(config, 'simulation', 'num_tasks', default='N/A')}")
    lines.append(f"arrival_rate: {_cfg(config, 'simulation', 'arrival_rate', default='N/A')}")
    lines.append(f"num_nodes: {_cfg(config, 'nodes', 'num_nodes', default='N/A')}")
    lines.append(f"schedulers: {config.get('schedulers', [])}")
    lines.append(f"fixed_window.window_size: {_cfg(config, 'fixed_window', 'window_size', default='N/A')}")
    lines.append(f"dynamic_window.min_batch_size: {_cfg(config, 'dynamic_window', 'min_batch_size', default='N/A')}")
    lines.append(f"dynamic_window.max_batch_size: {_cfg(config, 'dynamic_window', 'max_batch_size', default='N/A')}")
    lines.append(f"dynamic_window.min_wait: {_cfg(config, 'dynamic_window', 'min_wait', default='N/A')}")
    lines.append(f"dynamic_window.max_wait: {_cfg(config, 'dynamic_window', 'max_wait', default='N/A')}")
    lines.append(f"dynamic_window.slack_trigger: {_cfg(config, 'dynamic_window', 'slack_trigger', default='N/A')}")
    lines.append(f"dynamic_window.queue_pressure_trigger: {_cfg(config, 'dynamic_window', 'queue_pressure_trigger', default='N/A')}")
    lines.append(f"pruning.prune_time_factor: {_cfg(config, 'pruning', 'prune_time_factor', default='N/A')}")
    lines.append(f"pruning.queue_wait_threshold: {_cfg(config, 'pruning', 'queue_wait_threshold', default='N/A')}")
    lines.append(f"pruning.enable_topk: {_cfg(config, 'pruning', 'enable_topk', default='N/A')}")
    lines.append(f"pruning.top_k_candidates: {_cfg(config, 'pruning', 'top_k_candidates', default='N/A')}")
    lines.append(f"pruning.enable_energy_pruning: {_cfg(config, 'pruning', 'enable_energy_pruning', default='N/A')}")
    lines.append(f"pruning.energy_safety_factor: {_cfg(config, 'pruning', 'energy_safety_factor', default='N/A')}")
    lines.append(f"pruning.energy_wait_threshold: {_cfg(config, 'pruning', 'energy_wait_threshold', default='N/A')}")
    lines.append(f"energy.enable: {_cfg(config, 'energy', 'enable', default='N/A')}")
    lines.append(f"output_dir: {_cfg(config, 'output', 'output_dir', default='N/A')}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    summary_text = _build_config_summary(config, args.config)
    print(summary_text)

    config_snapshot_path = os.path.join(output_dir, "effective_config_snapshot.json")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    config_summary_txt_path = os.path.join(output_dir, "effective_config_summary.txt")
    with open(config_summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"Saved effective config snapshot to: {config_snapshot_path}")
    print(f"Saved effective config summary to: {config_summary_txt_path}")
    print()

    summary_rows = []
    schedulers = config.get("schedulers", ["greedy"])
    seeds = config.get("seeds", [1])

    for seed in seeds:
        set_global_seed(seed)

        nodes = generate_nodes(config, seed=seed)
        tasks = generate_tasks(config, nodes)

        print(f"[DEBUG] seed={seed} generated_tasks={len(tasks)} generated_nodes={len(nodes)}")
        if tasks:
            print(f"[DEBUG] seed={seed} Sample task: {tasks[0]}")
        if nodes:
            print(f"[DEBUG] seed={seed} Sample node: {nodes[0]}")
        print()

        for scheduler_name in schedulers:
            print(f"=== Running scheduler={scheduler_name}, seed={seed} ===")
            try:
                result = run_scheduler(scheduler_name, tasks, nodes, config)
            except NotImplementedError as e:
                print(f"[WARN] {e} Skipping.")
                print()
                continue
            except Exception as e:
                print(f"[ERROR] scheduler={scheduler_name}, seed={seed} failed: {e}")
                print()
                continue

            result["scheduler"] = scheduler_name
            result["seed"] = seed

            # Base defaults
            result.setdefault("num_dispatch_rounds", 0)
            result.setdefault("avg_window_size", 0.0)
            result.setdefault("candidate_pairs_before", 0)
            result.setdefault("candidate_pairs_after", 0)

            # Energy / pruning extended defaults
            result.setdefault("energy_pruned_pairs", 0)
            result.setdefault("time_pruned_pairs", 0)
            result.setdefault("queue_pruned_pairs", 0)
            result.setdefault("forced_energy_overdraw_count", 0)
            result.setdefault("avg_energy_wait_time", 0.0)
            result.setdefault("avg_remaining_energy", 0.0)
            result.setdefault("min_remaining_energy", 0.0)
            result.setdefault("total_remaining_energy", 0.0)
            result.setdefault("energy_enabled_nodes", 0)

            # Race defaults
            result.setdefault("base_branch_time", 0.0)
            result.setdefault("pruned_branch_time", 0.0)
            result.setdefault("race_finish_time", result.get("allocation_time", 0.0))
            result.setdefault("race_winner", "")
            result.setdefault("base_wins", 0)
            result.setdefault("pruned_wins", 0)

            result.setdefault("base_avg_completion_time", 0.0)
            result.setdefault("pruned_avg_completion_time", 0.0)
            result.setdefault("base_sla_violation_rate", 0.0)
            result.setdefault("pruned_sla_violation_rate", 0.0)
            result.setdefault("base_execution_time", 0.0)
            result.setdefault("pruned_execution_time", 0.0)

            result.setdefault("base_candidate_pairs_before", 0)
            result.setdefault("pruned_candidate_pairs_before", 0)
            result.setdefault("base_candidate_pairs_after", 0)
            result.setdefault("pruned_candidate_pairs_after", 0)

            result.setdefault("base_energy_pruned_pairs", 0)
            result.setdefault("pruned_energy_pruned_pairs", 0)
            result.setdefault("base_avg_energy_wait_time", 0.0)
            result.setdefault("pruned_avg_energy_wait_time", 0.0)
            result.setdefault("base_avg_remaining_energy", 0.0)
            result.setdefault("pruned_avg_remaining_energy", 0.0)

            result.setdefault("quality_gap_avg_completion_pruned_minus_base", 0.0)
            result.setdefault("quality_gap_sla_pruned_minus_base", 0.0)
            result.setdefault("quality_gap_exec_time_pruned_minus_base", 0.0)
            result.setdefault("quality_gap_energy_wait_pruned_minus_base", 0.0)

            before = result["candidate_pairs_before"]
            after = result["candidate_pairs_after"]

            if before > 0:
                result["candidate_keep_ratio"] = after / before
                result["candidate_prune_ratio"] = 1.0 - (after / before)
            else:
                result["candidate_keep_ratio"] = 1.0
                result["candidate_prune_ratio"] = 0.0

            print(f"Scheduler: {result['scheduler']}")
            print(f"Num tasks: {result['num_tasks']}")
            print(f"Average completion time: {result['avg_completion_time']:.6f}")
            print(f"SLA violation rate: {result['sla_violation_rate']:.6f}")
            print(f"Total allocation time: {result['allocation_time']:.6f}")
            print(f"Total execution time: {result['execution_time']:.12f}")
            print(f"Num dispatch rounds: {result['num_dispatch_rounds']}")
            print(f"Avg window size: {result['avg_window_size']:.6f}")
            print(f"Candidate pairs before pruning: {result['candidate_pairs_before']}")
            print(f"Candidate pairs after pruning: {result['candidate_pairs_after']}")
            print(f"Candidate keep ratio: {result['candidate_keep_ratio']:.6f}")
            print(f"Candidate prune ratio: {result['candidate_prune_ratio']:.6f}")
            print(f"Energy pruned pairs: {result['energy_pruned_pairs']}")
            print(f"Time pruned pairs: {result['time_pruned_pairs']}")
            print(f"Queue pruned pairs: {result['queue_pruned_pairs']}")
            print(f"Forced energy overdraw count: {result['forced_energy_overdraw_count']}")
            print(f"Average energy wait time: {result['avg_energy_wait_time']:.6f}")
            print(f"Average remaining energy: {result['avg_remaining_energy']:.6f}")
            print(f"Min remaining energy: {result['min_remaining_energy']:.6f}")

            if "race" in scheduler_name:
                print(f"Race winner: {result['race_winner']}")
                print(f"Base branch time: {result['base_branch_time']:.6f}")
                print(f"Pruned branch time: {result['pruned_branch_time']:.6f}")

            result_path = os.path.join(output_dir, f"{scheduler_name}_seed{seed}.json")
            save_json(result, result_path)
            print(f"Saved result to: {result_path}")
            print()

            summary_rows.append({
                "seed": seed,
                "scheduler": result["scheduler"],
                "num_tasks": result["num_tasks"],
                "avg_completion_time": result["avg_completion_time"],
                "sla_violation_rate": result["sla_violation_rate"],
                "allocation_time": result["allocation_time"],
                "execution_time": result["execution_time"],
                "num_dispatch_rounds": result["num_dispatch_rounds"],
                "avg_window_size": result["avg_window_size"],
                "candidate_pairs_before": result["candidate_pairs_before"],
                "candidate_pairs_after": result["candidate_pairs_after"],
                "candidate_keep_ratio": result["candidate_keep_ratio"],
                "candidate_prune_ratio": result["candidate_prune_ratio"],

                "energy_pruned_pairs": result["energy_pruned_pairs"],
                "time_pruned_pairs": result["time_pruned_pairs"],
                "queue_pruned_pairs": result["queue_pruned_pairs"],
                "forced_energy_overdraw_count": result["forced_energy_overdraw_count"],
                "avg_energy_wait_time": result["avg_energy_wait_time"],
                "avg_remaining_energy": result["avg_remaining_energy"],
                "min_remaining_energy": result["min_remaining_energy"],
                "total_remaining_energy": result["total_remaining_energy"],
                "energy_enabled_nodes": result["energy_enabled_nodes"],

                "base_branch_time": result["base_branch_time"],
                "pruned_branch_time": result["pruned_branch_time"],
                "race_finish_time": result["race_finish_time"],
                "race_winner": result["race_winner"],
                "base_wins": result["base_wins"],
                "pruned_wins": result["pruned_wins"],

                "base_avg_completion_time": result["base_avg_completion_time"],
                "pruned_avg_completion_time": result["pruned_avg_completion_time"],
                "base_sla_violation_rate": result["base_sla_violation_rate"],
                "pruned_sla_violation_rate": result["pruned_sla_violation_rate"],
                "base_execution_time": result["base_execution_time"],
                "pruned_execution_time": result["pruned_execution_time"],

                "base_candidate_pairs_before": result["base_candidate_pairs_before"],
                "pruned_candidate_pairs_before": result["pruned_candidate_pairs_before"],
                "base_candidate_pairs_after": result["base_candidate_pairs_after"],
                "pruned_candidate_pairs_after": result["pruned_candidate_pairs_after"],

                "base_energy_pruned_pairs": result["base_energy_pruned_pairs"],
                "pruned_energy_pruned_pairs": result["pruned_energy_pruned_pairs"],
                "base_avg_energy_wait_time": result["base_avg_energy_wait_time"],
                "pruned_avg_energy_wait_time": result["pruned_avg_energy_wait_time"],
                "base_avg_remaining_energy": result["base_avg_remaining_energy"],
                "pruned_avg_remaining_energy": result["pruned_avg_remaining_energy"],

                "quality_gap_avg_completion_pruned_minus_base": result["quality_gap_avg_completion_pruned_minus_base"],
                "quality_gap_sla_pruned_minus_base": result["quality_gap_sla_pruned_minus_base"],
                "quality_gap_exec_time_pruned_minus_base": result["quality_gap_exec_time_pruned_minus_base"],
                "quality_gap_energy_wait_pruned_minus_base": result["quality_gap_energy_wait_pruned_minus_base"],
            })

    summary_csv_path = os.path.join(output_dir, "all_summary.csv")
    save_csv(
        summary_rows,
        summary_csv_path,
        fieldnames=[
            "seed",
            "scheduler",
            "num_tasks",
            "avg_completion_time",
            "sla_violation_rate",
            "allocation_time",
            "execution_time",
            "num_dispatch_rounds",
            "avg_window_size",
            "candidate_pairs_before",
            "candidate_pairs_after",
            "candidate_keep_ratio",
            "candidate_prune_ratio",

            "energy_pruned_pairs",
            "time_pruned_pairs",
            "queue_pruned_pairs",
            "forced_energy_overdraw_count",
            "avg_energy_wait_time",
            "avg_remaining_energy",
            "min_remaining_energy",
            "total_remaining_energy",
            "energy_enabled_nodes",

            "base_branch_time",
            "pruned_branch_time",
            "race_finish_time",
            "race_winner",
            "base_wins",
            "pruned_wins",

            "base_avg_completion_time",
            "pruned_avg_completion_time",
            "base_sla_violation_rate",
            "pruned_sla_violation_rate",
            "base_execution_time",
            "pruned_execution_time",

            "base_candidate_pairs_before",
            "pruned_candidate_pairs_before",
            "base_candidate_pairs_after",
            "pruned_candidate_pairs_after",

            "base_energy_pruned_pairs",
            "pruned_energy_pruned_pairs",
            "base_avg_energy_wait_time",
            "pruned_avg_energy_wait_time",
            "base_avg_remaining_energy",
            "pruned_avg_remaining_energy",

            "quality_gap_avg_completion_pruned_minus_base",
            "quality_gap_sla_pruned_minus_base",
            "quality_gap_exec_time_pruned_minus_base",
            "quality_gap_energy_wait_pruned_minus_base",
        ],
    )

    print(f"Saved summary CSV to: {summary_csv_path}")
    print("All scheduler runs finished.")


if __name__ == "__main__":
    main()
