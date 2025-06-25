# MLP-Scheduling-Predictor_OSPROJ

This project uses a **Multi-Layer Perceptron (MLP)** neural network to predict the optimal **CPU scheduling algorithm** for a given set of processes based on their arrival and burst times.

It was developed as a final project for the **Operating Systems** course at Hamedan University of Technology (HUT).

---

## Project Goal

To automatically predict the scheduling algorithm with the **lowest total waiting time** among the following:

- `FCFS` (First Come First Serve)
- `SJF` (Shortest Job First)
- `RR` (Round Robin, with quantum = 4)

The model is trained using a dataset of simulated scheduling scenarios and implemented with **PyTorch**.

---

## Dataset

The dataset includes **1200 samples**, each consisting of:

- 4 process **arrival times**
- 4 process **burst times**
- A **label** indicating the best scheduling algorithm based on lowest total waiting time

### Label Selection Rules

In case of a tie in waiting times, the following priority is used:

- If `FCFS` and `SJF` tie → label is `FCFS`
- If `FCFS` and `RR` tie → label is `RR`
- If `SJF` and `RR` tie → label is `SJF`

---

## Model Architecture

- **Input Layer:** 8 features (4 arrival + 4 burst)
- **Hidden Layers:** 2 fully connected layers with ReLU activation and dropout
- **Output Layer:** 3 neurons (one for each scheduling algorithm)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

---

## Results

- **Test Accuracy:** 65.83%
- **Training Visualizations:** Plots of loss and accuracy over epochs using `matplotlib`

<p align="center">
  <img src="https://uploadkon.ir/uploads/89f425_25Figure-2.png" alt="Training Graphs" width="500"/>
</p>

---

##  How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Mahyas-G/MLP-Scheduling-Predictor_OSPROJ.git
   cd MLP-Scheduling-Predictor_OSPROJ
   ```

2. **(Optional) Create and activate virtual environment:**
   ```bash
   conda create -n mlp_sched python=3.9
   conda activate mlp_sched
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script:**
   ```bash
   python train.py
   ```

---

## Files Overview

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `dataset.csv`         | Final dataset with features and labels           |
| `dataset_generator.py`| Script to simulate processes and generate dataset|
| `model.py`            | PyTorch implementation of the MLP model          |
| `train.py`            | Training loop and evaluation logic               |
| `requirements.txt`    | List of required Python libraries                |
| `README.md`           | Project documentation (this file)                |

---

## Authors

- **Mahyas Golparian**  
- **Sara Kargar**

Final Project — Operating Systems Course, HUT  
Instructor: Dr. Mirhossein Dezfulian

---

## License
  
Feel free to use, modify, and share for educational purposes.

---
