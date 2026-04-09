from src.learning_baselines.minimal_gnn_scheduler import GNNScheduler

def run_gnn(tasks, nodes, config):
    scheduler = GNNScheduler(
        checkpoint_path=config.get("gnn", {}).get("checkpoint_path", "artifacts/minimal_gnn/minimal_gnn.pt"),
        task_dim=config.get("gnn", {}).get("task_dim", 5),
        server_dim=config.get("gnn", {}).get("server_dim", 5),
        pair_dim=config.get("gnn", {}).get("pair_dim", 5),
        hidden_dim=config.get("gnn", {}).get("hidden_dim", 64),
        emb_dim=config.get("gnn", {}).get("emb_dim", 32),
        dropout=config.get("gnn", {}).get("dropout", 0.0),
        device=config.get("gnn", {}).get("device", "cpu"),
        unique_server=config.get("gnn", {}).get("unique_server", True),
    )

    assignment, sched_info = scheduler.schedule(
        tasks=tasks,
        servers=nodes,
        current_time=0.0,
        env_state=None,
    )

    return {
        "scheduler": "gnn",
        "assignment": assignment,
        "sched_info": sched_info,
    }
