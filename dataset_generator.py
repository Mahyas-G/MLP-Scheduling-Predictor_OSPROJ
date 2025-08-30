import pandas as pd
import numpy as np
import random
from collections import Counter
import argparse
import os
from typing import List, Tuple

def fcfs_waiting_time(arrivals: List[int], bursts: List[int]) -> int:
    n = len(arrivals)
    start_time = 0
    waiting_times = []
    for i in range(n):
        if start_time < arrivals[i]:
            start_time = arrivals[i]
        waiting_times.append(start_time - arrivals[i])
        start_time += bursts[i]
    return int(sum(waiting_times))

def sjf_waiting_time(arrivals: List[int], bursts: List[int]) -> int:
    n = len(arrivals)
    processes = list(zip(arrivals, bursts))
    time = 0
    waiting_times = []
    ready = []
    completed = []
    while len(completed) < n:
        for i in range(n):
            if processes[i][0] <= time and i not in completed and i not in ready:
                ready.append(i)
        if not ready:
            time += 1
            continue
        shortest = min(ready, key=lambda i: processes[i][1])
        ready.remove(shortest)
        wait = time - processes[shortest][0]
        waiting_times.append(wait)
        time += processes[shortest][1]
        completed.append(shortest)
    return int(sum(waiting_times))

def rr_waiting_time(arrivals: List[int], bursts: List[int], quantum: int = 4) -> int:
    n = len(arrivals)
    remaining = bursts[:]
    waiting_times = [0] * n
    time = 0
    queue = []
    arrived = [False] * n
    done = 0
    while done < n:
        for i in range(n):
            if arrivals[i] <= time and not arrived[i]:
                queue.append(i)
                arrived[i] = True
        if queue:
            current = queue.pop(0)
            exec_time = min(quantum, remaining[current])
            remaining[current] -= exec_time
            time += exec_time
            for i in range(n):
                if i != current and arrivals[i] <= time and remaining[i] > 0:
                    waiting_times[i] += exec_time
            if remaining[current] > 0:
                queue.append(current)
            else:
                done += 1
        else:
            time += 1
    return int(sum(waiting_times))

def select_best_algorithm(fcfs_wt: int, sjf_wt: int, rr_wt: int) -> str:
    min_wt = min(fcfs_wt, sjf_wt, rr_wt)
    candidates = []
    if fcfs_wt == min_wt:
        candidates.append("FCFS")
    if sjf_wt == min_wt:
        candidates.append("SJF")
    if rr_wt == min_wt:
        candidates.append("RR")

    if len(candidates) > 1:
        if "FCFS" in candidates and "SJF" in candidates:
            return "FCFS"
        elif "SJF" in candidates and "RR" in candidates:
            return "SJF"
        elif "FCFS" in candidates and "RR" in candidates:
            return "RR"
        else:
            return candidates[0]
    return candidates[0]

def generate_one_sample(arr_low: int, arr_high: int, bt_low: int, bt_high: int, sort_arrivals: bool):
    arrivals = [random.randint(arr_low, arr_high) for _ in range(4)]
    if sort_arrivals:
        arrivals = sorted(arrivals)
    bursts = [random.randint(bt_low, bt_high) for _ in range(4)]

    fcfs_wt = fcfs_waiting_time(arrivals, bursts)
    sjf_wt = sjf_waiting_time(arrivals, bursts)
    rr_wt = rr_waiting_time(arrivals, bursts, quantum=4)
    label = select_best_algorithm(fcfs_wt, sjf_wt, rr_wt)

    row = arrivals + bursts + [label, fcfs_wt, sjf_wt, rr_wt]
    return row, label

def generate_dataset(
    output_path: str = "dataset.csv",
    balanced: bool = True,
    n_total: int = 1200,
    n_per_class: int = 400,
    arr_low: int = 0,
    arr_high: int = 10,
    bt_low: int = 1,
    bt_high: int = 10,
    sort_arrivals: bool = True,
    seed: int = None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    columns = [
        'P1_Arrival', 'P2_Arrival', 'P3_Arrival', 'P4_Arrival',
        'P1_Burst', 'P2_Burst', 'P3_Burst', 'P4_Burst',
        'Best_Algorithm', 'FCFS_WT', 'SJF_WT', 'RR_WT'
    ]

    data = []
    if not balanced:
        for _ in range(n_total):
            row, _ = generate_one_sample(arr_low, arr_high, bt_low, bt_high, sort_arrivals)
            data.append(row)
    else:
        target_counts = {'FCFS': n_per_class, 'SJF': n_per_class, 'RR': n_per_class}
        counts = Counter()
        max_attempts = sum(target_counts.values()) * 50
        attempts = 0

        while sum(counts.values()) < sum(target_counts.values()) and attempts < max_attempts:
            row, label = generate_one_sample(arr_low, arr_high, bt_low, bt_high, sort_arrivals)
            if counts[label] < target_counts[label]:
                data.append(row)
                counts[label] += 1
            attempts += 1

        for algo in target_counts:
            attempts_force = 0
            max_force_attempts = 200000
            while counts[algo] < target_counts[algo] and attempts_force < max_force_attempts:
                row, label = generate_one_sample(arr_low, arr_high, bt_low, bt_high, sort_arrivals)
                if label == algo:
                    data.append(row)
                    counts[algo] += 1
                attempts_force += 1

            if counts[algo] < target_counts[algo]:
                while counts[algo] < target_counts[algo]:
                    row, _ = generate_one_sample(arr_low, arr_high, bt_low, bt_high, sort_arrivals)
                    row[8] = algo
                    data.append(row)
                    counts[algo] += 1

    df = pd.DataFrame(data, columns=columns)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def parse_args():
    p = argparse.ArgumentParser(description="Dataset Generator (Balanced by default)")
    p.add_argument("--output", "-o", default="dataset.csv", help="Path to save the output CSV file (default: dataset.csv)")
    p.add_argument("--unbalanced", action="store_true", help="If set, generates an unbalanced dataset")
    p.add_argument("--n_total", type=int, default=1200, help="Total number of samples (only for unbalanced mode)")
    p.add_argument("--n_per_class", type=int, default=400, help="Number of samples per class (for balanced mode)")
    p.add_argument("--arr_low", type=int, default=0)
    p.add_argument("--arr_high", type=int, default=10)
    p.add_argument("--bt_low", type=int, default=1)
    p.add_argument("--bt_high", type=int, default=10)
    p.add_argument("--no_sort_arrivals", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sort_arrivals = not args.no_sort_arrivals
    df = generate_dataset(
        output_path=args.output,
        balanced=not args.unbalanced,
        n_total=args.n_total,
        n_per_class=args.n_per_class,
        arr_low=args.arr_low,
        arr_high=args.arr_high,
        bt_low=args.bt_low,
        bt_high=args.bt_high,
        sort_arrivals=sort_arrivals,
        seed=args.seed
    )
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Class counts: {df['Best_Algorithm'].value_counts().to_dict()}")
