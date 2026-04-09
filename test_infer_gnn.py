from src.learning_baselines.minimal_gnn_scheduler import GNNScheduler


def main():
    tasks = [
        {
            "task_id": 0,
            "required_compute": 30.0,
            "input_size": 2.0,
            "deadline": 5.0,
            "arrival_time": 0.0,
            "priority": 1.0,
        },
        {
            "task_id": 1,
            "required_compute": 60.0,
            "input_size": 4.0,
            "deadline": 8.0,
            "arrival_time": 0.2,
            "priority": 1.0,
        },
    ]

    servers = [
        {
            "node_id": 0,
            "compute_capacity": 50.0,
            "queue_length": 0.0,
            "remaining_energy": 100.0,
        },
        {
            "node_id": 1,
            "compute_capacity": 80.0,
            "queue_length": 1.0,
            "remaining_energy": 120.0,
        },
        {
            "node_id": 2,
            "compute_capacity": 30.0,
            "queue_length": 0.0,
            "remaining_energy": 60.0,
        },
    ]

    scheduler = GNNScheduler(
        checkpoint_path="artifacts/minimal_gnn/minimal_gnn.pt",
        task_dim=5,
        server_dim=5,
        pair_dim=5,
        hidden_dim=64,
        emb_dim=32,
        dropout=0.0,
        device="cpu",
        unique_server=True,
    )

    result = scheduler.schedule(
        tasks=tasks,
        servers=servers,
        current_time=0.0,
        env_state=None,
    )

    print("result =", result)

    if isinstance(result, tuple) and len(result) == 2:
        assignment, info = result
        print("assignment =", assignment)
        print("info =", info)


if __name__ == "__main__":
    main()
