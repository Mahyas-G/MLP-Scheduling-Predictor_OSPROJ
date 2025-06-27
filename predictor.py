import torch
import os
import glob
from model import MLPModel

model_files = glob.glob("assets/model_*.pth")

if not model_files:
    print("Error: No trained model found in 'assets/' folder.")
    print("Please run 'train.py' first to generate a model.")
    exit()

latest_model = max(model_files, key=os.path.getctime)
print(f"Loading model from: {latest_model}")

model = MLPModel()
model.load_state_dict(torch.load(latest_model))
model.eval()

label_map = {0: "FCFS", 1: "SJF", 2: "RR"}

def get_valid_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Value must be non-negative. Try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    print("\nEnter arrival and burst times for 4 processes:")
    arrival_times = []
    burst_times = []

    for i in range(4):
        print(f"\n Process P{i+1}:")
        at = get_valid_input("   Arrival time: ")
        bt = get_valid_input("   Burst time: ")
        arrival_times.append(at)
        burst_times.append(bt)

    features = arrival_times + burst_times
    X = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        output = model(X)
        _, pred = torch.max(output.data, 1)
        predicted_label = pred.item()

    print(f"\nRecommended Scheduling Algorithm: {label_map[predicted_label]}\n")

if __name__ == "__main__":
    main()
