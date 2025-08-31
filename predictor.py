import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import MLPModel

try:
    import joblib
except ImportError:
    joblib = None

def get_latest(patterns):
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)

def load_label_map():
    label_json = get_latest(["assets/label2idx_*.json", "assets/labels_*.json", "assets/label_map_*.json"])
    if label_json:
        with open(label_json, "r", encoding="utf-8") as f:
            label2idx = json.load(f)
        idx2label = {int(v): k for k, v in label2idx.items()}
        return idx2label, label_json
    return {0: "FCFS", 1: "SJF", 2: "RR"}, None

def ask_float(prompt):
    while True:
        try:
            val = float(input(prompt))
            if val < 0:
                print("Value must be non-negative. Try again.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser(description="Predict the best CPU scheduling algorithm for 4 processes.")
    parser.add_argument("--no_sort_arrivals", action="store_true", help="Disable sorting of arrival times.")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth model file.")
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler .pkl file.")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu or cuda.")
    parser.add_argument("--arrivals", type=float, nargs=4, default=None, help="Four arrival times for P1..P4.")
    parser.add_argument("--bursts", type=float, nargs=4, default=None, help="Four burst times for P1..P4.")
    parser.add_argument("--show_probs", action="store_true", help="Show class probabilities.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    model_path = args.model or get_latest(["assets/best_model_*.pth", "assets/model_*.pth"])
    if not model_path:
        print("Error: No model found in assets/. Please run train.py first.")
        return
    print(f"Loading model from: {model_path}")

    model = MLPModel()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if args.scaler:
        scaler_path = args.scaler
    else:
        scaler_path = get_latest(["assets/scaler_*.pkl", "assets/standard_scaler_*.pkl"])
    if scaler_path is None:
        print("Warning: No scaler found in assets/. Predictions may be inaccurate.")
        scaler = None
    else:
        if joblib is None:
            print("Error: joblib is required to load scaler. Install with: pip install joblib")
            return
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from: {scaler_path}")

    idx2label, label_json = load_label_map()
    if label_json:
        print(f"Loaded label map from: {label_json}")
    else:
        print("Using default label map: {0:'FCFS', 1:'SJF', 2:'RR'}")

    if args.arrivals is not None and args.bursts is not None:
        arrival_times = list(args.arrivals)
        burst_times = list(args.bursts)
        if len(arrival_times) != 4 or len(burst_times) != 4 or any(a < 0 for a in arrival_times+burst_times):
            print("Invalid input. You must provide 4 non-negative arrival and 4 non-negative burst times.")
            return
    else:
        print("\nEnter arrival and burst times for 4 processes:")
        arrival_times = []
        burst_times = []
        for i in range(4):
            print(f"\nProcess P{i+1}:")
            at = ask_float("  Arrival time: ")
            bt = ask_float("  Burst time: ")
            arrival_times.append(at)
            burst_times.append(bt)

    sort_arrivals = not args.no_sort_arrivals
    if sort_arrivals:
        arrival_times = sorted(arrival_times)

    features = np.array(arrival_times + burst_times, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)

    X = torch.tensor(features, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    label_name = idx2label.get(pred_idx, f"CLASS_{pred_idx}")

    print("\n==== Prediction ====")
    print(f"Recommended Scheduling Algorithm: {label_name}")

    if args.show_probs:
        order = np.argsort(-probs)
        print("\nClass Probabilities:")
        for k in order:
            print(f"  {idx2label.get(int(k), f'CLASS_{int(k)}')}: {probs[k]:.4f}")

if __name__ == "__main__":
    main()
