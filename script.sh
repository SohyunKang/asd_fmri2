#!/bin/bash

GPU=3

python3 worker.py --worker_id 0 --n_workers 4 --gpu $GPU > log_worker0.txt 2>&1 &
python3 worker.py --worker_id 1 --n_workers 4 --gpu $GPU > log_worker1.txt 2>&1 &
python3 worker.py --worker_id 2 --n_workers 4 --gpu $GPU > log_worker2.txt 2>&1 &
python3 worker.py --worker_id 3 --n_workers 4 --gpu $GPU > log_worker3.txt 2>&1 &

wait

echo "All workers finished."