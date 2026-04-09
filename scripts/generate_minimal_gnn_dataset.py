# scripts/generate_minimal_gnn_dataset.py

import random

from src.learning_baselines.minimal_gnn_dataset import create_sample, save_samples


def get_training_states_from_simulator():
    """
    你需要替换这里：
    返回一个可迭代对象，每个元素形如:
    {
        "tasks": [...],
        "servers": [...],
        "current_time": 123.0,
    }
    """
    raise NotImplementedError("Please connect this to your simulator states.")


def run_teacher_scheduler(tasks, servers, current_time):
    """
    你需要替换这里：
    返回 teacher assignment, 格式建议:
      {task_idx: server_idx}
    teacher 建议用:
      dynamic_window_pruned
      或 dynamic_window_race
    """
    raise NotImplementedError("Please connect this to your teacher scheduler.")


def main():
    states = list(get_training_states_from_simulator())
    random.shuffle(states)

    samples = []
    for st in states:
        tasks = st["tasks"]
        servers = st["servers"]
        current_time = st["current_time"]

        teacher_assignment = run_teacher_scheduler(tasks, servers, current_time)
        sample = create_sample(tasks, servers, current_time, teacher_assignment)
        samples.append(sample)

    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    save_samples(train_samples, "artifacts/minimal_gnn/train.pkl")
    save_samples(val_samples, "artifacts/minimal_gnn/val.pkl")
    save_samples(test_samples, "artifacts/minimal_gnn/test.pkl")

    print(f"saved train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")


if __name__ == "__main__":
    main()
