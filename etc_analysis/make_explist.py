import itertools
import json
import random

N_TRIALS = 1000
SEED = 42

search_space = {
    "d_model": [96, 128],
    "n_hops": [5, 6, 7],
    "depth": [2, 3],
    "top_m": [6, 8, 10],
    "seed_std": [0.3, 0.5, 0.7],
    "mlp_hidden": [128, 256],
    "dropout": [0.05, 0.1, 0.2],
    "lr": [1e-4, 3e-4, 5e-4],
    "seed_lr": [1e-3, 3e-3, 1e-2],
    "batch_size": [8, 16, 32, 64, 128],
}

keys = list(search_space.keys())
all_configs = [
    dict(zip(keys, values))
    for values in itertools.product(*(search_space[k] for k in keys))
]

random.seed(SEED)
random.shuffle(all_configs)

configs = all_configs[200:N_TRIALS]
# configs = all_configs[N_TRIALS:2*N_TRIALS]

with open("search_configs2.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Saved {len(configs)} configs to search_configs.json")