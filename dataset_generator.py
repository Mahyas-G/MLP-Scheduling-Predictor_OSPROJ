import pandas as pd
import numpy as np

np.random.seed(42)

samples = []
algorithms = ['FCFS', 'SJF', 'RR']

def fcfs(arrival, burst):
    order = np.argsort(arrival)
    time = 0
    waiting = []
    for i in order:
        time = max(time, arrival[i])
        waiting.append(time - arrival[i])
        time += burst[i]
    return sum(waiting)

def sjf(arrival, burst):
    processes = list(range(4))
    time = 0
    waiting = []
    ready = []
    remaining = processes.copy()
    
    while remaining or ready:
        for i in remaining[:]:
            if arrival[i] <= time:
                ready.append(i)
                remaining.remove(i)
        if ready:
            ready.sort(key=lambda x: burst[x])
            i = ready.pop(0)
            waiting.append(time - arrival[i])
            time = max(time, arrival[i]) + burst[i]
        else:
            time += 1
    return sum(waiting)

def rr(arrival, burst, quantum=4):
    n = 4
    remaining_burst = burst.copy()
    time = 0
    waiting = [0] * n
    last_executed = [-1] * n
    completed = [False] * n
    queue = []
    arrival_times = arrival.copy()
    processes = list(range(n))

    while not all(completed):
        for i in processes:
            if arrival[i] <= time and i not in queue and remaining_burst[i] > 0:
                queue.append(i)
        if queue:
            i = queue.pop(0)
            if last_executed[i] == -1:
                waiting[i] = time - arrival[i]
            else:
                waiting[i] += time - last_executed[i]
            exec_time = min(quantum, remaining_burst[i])
            time += exec_time
            remaining_burst[i] -= exec_time
            last_executed[i] = time
            for j in processes:
                if arrival[j] <= time and j not in queue and remaining_burst[j] > 0:
                    queue.append(j)
            if remaining_burst[i] > 0:
                queue.append(i)
            else:
                completed[i] = True
        else:
            time += 1
    return sum(waiting)

for _ in range(1200):
    arrival = np.random.randint(0, 20, size=4)
    burst = np.random.randint(1, 10, size=4)
    
    wt_fcfs = fcfs(arrival, burst)
    wt_sjf = sjf(arrival, burst)
    wt_rr = rr(arrival, burst)
    
    if wt_fcfs <= wt_sjf and wt_fcfs <= wt_rr:
        label = 'FCFS'
    elif wt_sjf <= wt_fcfs and wt_sjf <= wt_rr:
        label = 'SJF'
    else:
        label = 'RR'
    
    row = list(arrival) + list(burst) + [label]
    samples.append(row)

columns = [f'Arrival_{i+1}' for i in range(4)] + [f'Burst_{i+1}' for i in range(4)] + ['Best_Algorithm']
df = pd.DataFrame(samples, columns=columns)
df.to_csv('dataset.csv', index=False)
print("Dataset generated and saved to dataset.csv")
