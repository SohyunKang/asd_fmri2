import subprocess
import time

GPU = 3
SEEDS = list(range(100))   # 0~99
MAX_PARALLEL = 2           # GPU 상황에 맞게 (2 추천)

running = []

for seed in SEEDS:

    # 동시에 너무 많이 돌지 않게 제한
    while len(running) >= MAX_PARALLEL:
        for p, f in running:
            if p.poll() is not None:
                f.close()
                running.remove((p, f))
        time.sleep(1)

    cmd = [
        "python3", "main.py",
        "--d_model", "96",
        "--n_hops", "5",
        "--depth", "3",
        "--top_m", "10",
        "--seed_std", "0.5",
        "--mlp_hidden", "256",
        "--dropout", "0.5",
        "--lr", "0.0005",
        "--seed_lr", "0.001",
        "--weight_decay", "0.0001",
        "--batch_size", "16",
        "--epochs", "200",

        # 🔥 seed만 변경
        "--seed", str(seed),
    ]

    log_file = f"log_seed{seed}.txt"
    f = open(log_file, "w")

    print(f"🚀 Running seed {seed}")

    p = subprocess.Popen(
        f"CUDA_VISIBLE_DEVICES={GPU} " + " ".join(cmd),
        shell=True,
        stdout=f,
        stderr=f
    )

    running.append((p, f))

# 마지막까지 대기
for p, f in running:
    p.wait()
    f.close()

print("✅ All seeds finished")