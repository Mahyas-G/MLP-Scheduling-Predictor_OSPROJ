import pandas as pd
import numpy as np
import random

def fcfs_waiting_time(arrivals, bursts):
    n = len(arrivals)
    start_time = 0
    waiting_times = []
    for i in range(n):
        if start_time < arrivals[i]:
            start_time = arrivals[i]
        waiting_times.append(start_time - arrivals[i])
        start_time += bursts[i]
    return sum(waiting_times)

def sjf_waiting_time(arrivals, bursts):
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

    return sum(waiting_times)

def rr_waiting_time(arrivals, bursts, quantum=4):
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

    return sum(waiting_times)

def select_best_algorithm(arrivals, bursts):
    wt_fcfs = fcfs_waiting_time(arrivals, bursts)
    wt_sjf = sjf_waiting_time(arrivals, bursts)
    wt_rr = rr_waiting_time(arrivals, bursts)

    min_wt = min(wt_fcfs, wt_sjf, wt_rr)

    if min_wt == wt_fcfs:
        return "FCFS"
    elif min_wt == wt_sjf:
        return "SJF"
    else:
        return "RR"

data = []
for _ in range(1200): 
    arrivals = sorted([random.randint(0, 10) for _ in range(4)])
    bursts = [random.randint(1, 10) for _ in range(4)]
    label = select_best_algorithm(arrivals, bursts)
    row = arrivals + bursts + [label]
    data.append(row)

columns = ['P1_Arrival', 'P2_Arrival', 'P3_Arrival', 'P4_Arrival',
           'P1_Burst', 'P2_Burst', 'P3_Burst', 'P4_Burst',
           'Best_Algorithm']

df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)
print("Dataset saved as dataset.csv with 1200 samples.")
