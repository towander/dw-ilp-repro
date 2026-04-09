import random


def generate_nodes(config, seed=1):
    """
    Generate edge nodes with a local RNG, so we do NOT reset global random state.

    Energy-aware extension:
    - battery_capacity
    - battery_level
    - harvest_rate
    - min_reserve_energy
    - tx_energy_coeff
    """
    rng = random.Random(seed + 1000)

    node_cfg = config["nodes"]
    energy_cfg = config.get("energy", {})

    num_nodes = node_cfg["num_nodes"]

    capacity_min = int(node_cfg.get("compute_capacity_min", 20))
    capacity_max = int(node_cfg.get("compute_capacity_max", 80))

    bandwidth_min = float(node_cfg.get("bandwidth_min", 10.0))
    bandwidth_max = float(node_cfg.get("bandwidth_max", 50.0))

    power_min = float(node_cfg.get("power_coeff_min", 0.8))
    power_max = float(node_cfg.get("power_coeff_max", 1.5))

    energy_enable = bool(energy_cfg.get("enable", False))

    battery_capacity_min = float(energy_cfg.get("battery_capacity_min", 80.0))
    battery_capacity_max = float(energy_cfg.get("battery_capacity_max", 150.0))
    initial_ratio_min = float(energy_cfg.get("initial_battery_ratio_min", 0.45))
    initial_ratio_max = float(energy_cfg.get("initial_battery_ratio_max", 0.90))
    harvest_rate_min = float(energy_cfg.get("harvest_rate_min", 0.05))
    harvest_rate_max = float(energy_cfg.get("harvest_rate_max", 0.30))
    min_reserve_energy = float(energy_cfg.get("min_reserve_energy", 8.0))
    tx_energy_coeff = float(energy_cfg.get("tx_energy_coeff", 0.03))

    nodes = []
    for i in range(num_nodes):
        if energy_enable:
            battery_capacity = rng.uniform(battery_capacity_min, battery_capacity_max)
            init_ratio = rng.uniform(initial_ratio_min, initial_ratio_max)
            battery_level = battery_capacity * init_ratio
            harvest_rate = rng.uniform(harvest_rate_min, harvest_rate_max)
            reserve_energy = min_reserve_energy
        else:
            battery_capacity = float("inf")
            battery_level = float("inf")
            harvest_rate = 0.0
            reserve_energy = 0.0

        node = {
            "node_id": i,
            "compute_capacity": rng.randint(capacity_min, capacity_max),
            "bandwidth": rng.uniform(bandwidth_min, bandwidth_max),
            "parallelism": rng.choice([1, 2, 4]),
            "power_coeff": rng.uniform(power_min, power_max),

            # energy-aware fields
            "energy_enabled": energy_enable,
            "battery_capacity": battery_capacity,
            "battery_level": battery_level,
            "harvest_rate": harvest_rate,
            "min_reserve_energy": reserve_energy,
            "tx_energy_coeff": tx_energy_coeff,
        }
        nodes.append(node)

    return nodes
