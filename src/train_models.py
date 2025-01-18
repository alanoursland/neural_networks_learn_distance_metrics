import gc
import json
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
import models

dir_exists_ok = True
runs_per_model = 20
seed = 42484103

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print("No GPU found. Exiting...")
    sys.exit(1)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Move entire datasets to GPU
X_train = train_dataset.data.float().to(device)
y_train = train_dataset.targets.to(device)
X_test = test_dataset.data.float().to(device)
y_test = test_dataset.targets.to(device)

# Reshape data
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def json_save(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f)

def evaluate_model(model, data, target, criterion = None):
    outputs = model(data)
    predict = torch.argmax(outputs, dim=1)
    acc = (predict == target).sum().item() / len(target) * 100
    error = criterion(outputs, target) if criterion is not None else None

    return acc, error

def train_full_dataset(model, dir_results, epochs=10, lr=0.1):
    """Train using entire dataset for each epoch"""
    model.train()

    checkpoint_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    next_threshold = checkpoint_thresholds[0]
    crossed_thresholds = set()
    
    # Setup metrics tracking
    epoch_data = {
        'loss': [],
        'train_acc': [],
        'test_acc': [],
        'timestamp': []
    }
    
    # Create directories for saving results
    dir_checkpoint = dir_results / "checkpoints"
    dir_checkpoint.mkdir(exist_ok=dir_exists_ok)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
        
    # Calculate initial accuracies
    with torch.no_grad():
        train_acc, train_loss = evaluate_model(model, X_train, y_train, criterion)
        test_acc, _ = evaluate_model(model, X_test, y_test)

    # Save initial metrics
    epoch_data['loss'].append(train_loss.item())
    epoch_data['train_acc'].append(train_acc)
    epoch_data['test_acc'].append(test_acc)
    epoch_data['timestamp'].append(time.time() - start_time)

    print(f'Epoch {0}/{epochs}: Train: {train_acc:.2f} Test:{test_acc:.2f}')
    
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        train_acc, train_loss = evaluate_model(model, X_train, y_train, criterion=criterion)
        train_loss.backward()
        optimizer.step()

        # nothing in these models is affected by eval/train modes so don't switch
        with torch.no_grad():
            test_acc, _ = evaluate_model(model, X_test, y_test)

        # Print progress every 500 epochs
        if epoch % 500 == 0 or epoch == epochs:
            print(f'Epoch {epoch}/{epochs}: Train: {train_acc:.2f} Test:{test_acc:.2f}')

        # Save metrics
        epoch_data['loss'].append(train_loss.item())
        epoch_data['train_acc'].append(train_acc)
        epoch_data['test_acc'].append(test_acc)
        epoch_data['timestamp'].append(time.time() - start_time)

        # Save checkpoint if we cross a threshold
        if train_acc > next_threshold:
            chk_json_file = dir_checkpoint / f"{epoch}.json"
            chk_model_file = dir_checkpoint / f"{epoch}_model.pt"
            chk_opt_file = dir_checkpoint / f"{epoch}_opt.pt"

            json_save({
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                }, chk_json_file)
            torch.save(model, chk_model_file)
            torch.save(optimizer, chk_opt_file)
            crossed_thresholds.add(next_threshold)

            # find the next threshold
            for i, threshold in enumerate(checkpoint_thresholds):
                if train_acc > threshold:
                    next_threshold = checkpoint_thresholds[i+1]

        # scheduler.step()

    # Save final model
    json_save(epoch_data, dir_results / "metrics.json")
    torch.save(model, dir_results / f"{model.name()}.pt")
    torch.save(optimizer, dir_results / "optimizer.pt")

    return epoch_data

def evaluate_all_models(model_list, runs_per_model, epochs=5000, lr=0.001, exp_name="models"):
    # Create experiment directory and save config
    dir_results = Path(f"results")
    dir_exp = dir_results / exp_name
    dir_exp.mkdir(parents=True, exist_ok=dir_exists_ok)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # For reproducibility, ensure that CUDA operations are deterministic (can slow down computation)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Save experiment config
    exp_config = {
        "learning_rate": lr,
        "criterion": "CrossEntropyLoss",
        "optimizer": "SGD",
        "epochs": epochs,
        "runs_per_model": runs_per_model,
        "batch_size": "full",
        "date": time.strftime("%Y-%m-%d"),
        "dataset": "MNIST",
        "data_transform": "Normalize((0.1307,), (0.3081,))",
        "seed": seed,
    }
    with open(dir_exp / "experiment_config.json", 'w') as f:
            json.dump(exp_config, f, indent=4)

    for model in model_list:
        model_name = model.name()
        dir_model  = dir_exp / model_name
        dir_model.mkdir(parents=True, exist_ok=dir_exists_ok)
        model_config = {
            "name": model_name,
            "description": model.description(),
            "architecture": str(model)
        }
        with open(dir_model / "model_config.json", 'w') as f:
            json.dump(model_config, f, indent=4)

        for run_i in range(runs_per_model):
            print(f"\nStarting {model.name()} - {model.description()} - Run {run_i + 1}/{runs_per_model}")

            dir_run = dir_model / str(run_i)
            dir_run.mkdir(parents=True, exist_ok=dir_exists_ok)

            start_time = time.time()
            
            # Reset model weights for new run
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

            train_full_dataset(model, dir_run, epochs=epochs, lr=lr)
            print(f"Full dataset training time: {time.time() - start_time:.2f} seconds\n")
            
            clear_gpu_memory()

# Example usage:
if __name__ == "__main__":
    # Initialize model
    model_list = models.create_model_list(device)
    evaluate_all_models(model_list, runs_per_model=runs_per_model)

"""
Directory structure

results/
  experiment_name/
    experiment_config.json
    ModelName/
      model_config.json
      0/
        metrics.json
        ModelName.pt
        opt.pt
        checkpoints/
          {epoch}.json
          {epoch}_model.pt
          {epoch}_opt.pt
      1/
      2/
"""

"""
The generated directory structure and files include:

- **`results/`**: Root directory for storing experiment results.
  - **`experiment_name/`** (e.g., `"models"`): Directory for each experiment.
    - **`experiment_config.json`**: JSON file with experiment configuration (learning rate, criterion, optimizer, epochs, runs per model, batch size, dataset, data transform, and seed).
    - **`ModelName/`** (e.g., `"ReLU"`, `"Abs"`): Directory for each model variant.
      - **`model_config.json`**: JSON file describing the model (name, architecture, description).
      - **`0/`, `1/`, `2/`...**: Directories for each run of the model.
        - **`metrics.json`**: JSON file storing metrics (loss, train accuracy, test accuracy, timestamp per epoch).
        - **`ModelName.pt`**: Model weights file.
        - **`opt.pt`**: Optimizer state file.
        - **`checkpoints/`**: Directory for checkpoint files.
          - **`{epoch}.json`**: JSON checkpoint with training accuracy and test accuracy at given epoch.
          - **`{epoch}_model.pt`**: Model checkpoint weights.
          - **`{epoch}_opt.pt`**: Optimizer checkpoint state.

"""