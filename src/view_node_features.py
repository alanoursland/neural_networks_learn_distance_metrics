import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from collections import defaultdict
import os
import json
from scipy.stats import entropy
import models

def calculate_node_entropy(node_data, targets, num_bins):
    """Calculate entropy for a single node's output distribution."""
    # Get global range for consistent binning
    data_min, data_max = node_data.min(), node_data.max()
    bins = np.linspace(data_min, data_max, num_bins + 1)
    
    # Calculate histogram for each class
    class_distributions = []
    total_samples = len(node_data)
    
    for i in range(10):  # 10 MNIST classes
        class_data = node_data[targets == i]
        hist, _ = np.histogram(class_data, bins=bins)
        class_distributions.append(hist)
    
    # Convert to numpy array for easier manipulation
    class_distributions = np.array(class_distributions)  # shape: (10, num_bins)
    
    # Calculate total samples in each bin
    bin_totals = np.sum(class_distributions, axis=0)  # shape: (num_bins,)
    bin_weights = bin_totals / total_samples  # normalize to get weights
    
    # Calculate entropy across classes for each bin
    bin_entropies = []
    for bin_idx in range(num_bins):
        bin_dist = class_distributions[:, bin_idx]
        # Only calculate entropy if bin has data
        if np.sum(bin_dist) > 0:
            bin_dist = bin_dist / np.sum(bin_dist)
            bin_entropies.append(entropy(bin_dist))
        else:
            bin_entropies.append(0)
    
    # Calculate weighted average entropy
    weighted_entropy = np.sum(np.array(bin_entropies) * bin_weights)
    
    return weighted_entropy

def plot_layer_histogram(layer_data, targets, node_idx, num_bins, title, output_path, use_log_y=False, count_zeros=False):
    """Plot and save separate histograms for each class for a single node."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns for 10 classes
    fig.suptitle(f'{title} - Node {node_idx}', fontsize=16)
    
    # Get data for this node
    node_data = layer_data[:,node_idx]
    
    # Add zero count if requested
    if count_zeros:
        zero_count = np.sum(np.isclose(node_data, 0.0, atol=1e-10))
        zero_percentage = (zero_count / len(node_data)) * 100
        plt.figtext(0.5, 0.95, f'Zero values: {zero_count:,} ({zero_percentage:.2f}%)', 
                    ha='center', fontsize=10)
    
    # Calculate bins based on node min/max 
    layer_min = node_data.min()
    layer_max = node_data.max()
    bins = np.linspace(layer_min, layer_max, num_bins + 1)
    
    # Colors for the 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Create separate histogram for each class
    for i in range(10):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Get data for this class
        class_data = node_data[targets == i]
        
        # Add class-specific zero count if requested
        title_text = f'Class {i}'
        if count_zeros:
            class_zeros = np.sum(np.isclose(class_data, 0.0, atol=1e-10))
            class_zero_percentage = (class_zeros / len(class_data)) * 100
            title_text += f'\nZeros: {class_zeros:,} ({class_zero_percentage:.2f}%)'
        
        # Plot histogram
        ax.hist(class_data, bins=bins, color=colors[i], alpha=0.7)

        # Set log scale if requested
        if use_log_y:
            ax.set_yscale('log')
            # Add small offset to avoid log(0)
            ax.set_ylim(bottom=0.1)

        ax.set_title(title_text)
        ax.set_xlabel('Node Output')
        ax.set_ylabel('Count (log scale)' if use_log_y else 'Count')
    
        # Add grid for easier reading of log scale
        if use_log_y:
            ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_projection(layer_data, targets, node_idx, num_bins, title, output_path, use_log_y=False, count_zeros=False):
    """Plot and save combined histograms for all classes for a single node on one number line."""
    # Get data for this node
    node_data = layer_data[:, node_idx]

    # Add zero count if requested
    if count_zeros:
        zero_count = np.sum(np.isclose(node_data, 0.0, atol=1e-10))
        zero_percentage = (zero_count / len(node_data)) * 100
        zero_info = f'Zero values: {zero_count:,} ({zero_percentage:.2f}%)'
    else:
        zero_info = ''

    # Calculate bins based on node min/max
    layer_min = node_data.min()
    layer_max = node_data.max()
    bins = np.linspace(layer_min, layer_max, num_bins + 1)

    # Colors for the 10 classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create a figure
    plt.figure(figsize=(12, 6))
    plt.title(f'{title} - Node {node_idx}', fontsize=16)

    # Plot histograms for each class on the same axis
    for i in range(10):
        class_data = node_data[targets == i]
        plt.hist(class_data, bins=bins, color=colors[i], alpha=0.5, label=f'Class {i}', histtype='stepfilled')

    # Set log scale if requested
    if use_log_y:
        plt.yscale('log')
        plt.ylim(bottom=0.1)  # Avoid log(0)

    # Add labels and legend
    plt.xlabel('Node Output')
    plt.ylabel('Count (log scale)' if use_log_y else 'Count')
    plt.legend(loc='upper right', title='Classes')
    plt.figtext(0.5, 0.01, zero_info, ha='center', fontsize=10)
    plt.grid(alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_layer(layer_data, targets, layer_name, output_dir, num_bins, use_log_y=False, count_zeros=False):
    """Process a layer: create histograms and calculate entropies."""
    num_nodes = layer_data.shape[1]
    entropies = {}
    zero_counts = {} if count_zeros else None
    
    for node in range(num_nodes):
        print(f"Processing node {node}/{num_nodes}")
        # Create histogram
        output_path = os.path.join(output_dir, f'{layer_name}_node_{node:03d}.png')
        plot_layer_histogram(
            layer_data, targets, node, num_bins,
            f'{layer_name} Node Outputs', output_path, 
            use_log_y=use_log_y,
            count_zeros=count_zeros
        )
        
        # Calculate entropy
        node_entropy = calculate_node_entropy(layer_data[:, node], targets, num_bins)
        entropies[f'node_{node:03d}'] = float(node_entropy)
        
        # Calculate zero counts if requested
        if count_zeros:
            node_data = layer_data[:, node]
            zero_count = np.sum(np.isclose(node_data, 0.0, atol=1e-10))
            zero_percentage = (zero_count / len(node_data)) * 100
            zero_counts[f'node_{node:03d}'] = {
                'count': int(zero_count),
                'percentage': float(zero_percentage)
            }
    
    return entropies, zero_counts

def get_layers_to_analyze(model):
    """Get the relevant layers to analyze and assign appropriate names and titles based on their type."""
    layers_to_analyze = []
    model_name = model.name()

    for name, module in model.named_modules():
        if not name:
            continue
        if type(module).__name__ == "PerturbationLayer":
            continue
        if type(module).__name__ == "Sequential":
            continue
        layer_desc = f"{model_name}_{name}"
        title = f"{model_name} - {type(module).__name__} ({name})"
        layers_to_analyze.append((layer_desc, module, title))

    # Print discovered layers for debugging
    print("\nDiscovered layers:")
    for name, layer, title in layers_to_analyze:
        print(f"- Name: {name},  Type:{type(layer).__name__}, Title: {title}")

    return layers_to_analyze
    
def main():
    parser = argparse.ArgumentParser(description='Analyze neural network node outputs')
    parser.add_argument('model_path', type=str, help='Path to the .pt model file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='Number of bins for histograms')
    parser.add_argument('--use_log_y', type=bool, default=False,
                        help='Does bin count use log')
    parser.add_argument('--count_zeros', action='store_true',
                        help='Count and display zero values in histograms')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("No GPU found. Exiting...")
        sys.exit(1)

    # Load model
    print("Loading Model")
    try:
        model = torch.load(args.model_path)
        model = model.to(device)
        model.eval()
        print(f"Loaded model: {model.name()}")
        print(f"Architecture: {model.description()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load and prepare MNIST test data
    print("Loading Data")
    test_dataset = datasets.MNIST('./data', train=False, download=True)
    X_test = test_dataset.data.float().to(device)
    y_test = test_dataset.targets.to(device)
    
    X_test = X_test.reshape(-1, 28*28)
    X_test = (X_test - 0.1307) / 0.3081

    # Storage for layer outputs
    layer_outputs = {}

    # Find Linear layers
    print("Introspecting model layers")
    layers_to_analyze = get_layers_to_analyze(model)
    
    if len(layers_to_analyze) < 2:
        print(f"Expected at least 2 layers, found {len(layers_to_analyze)}. Exiting...")
        sys.exit(1)

    # Create hooks
    print("Creating model hooks")

    print(layers_to_analyze)
    hooks = []
    for idx, (name, layer, description) in enumerate(layers_to_analyze):
        def make_hook(layer_name, layer_idx):
            def hook(module, input, output):
                layer_outputs[f'{layer_name}_{layer_idx}'] = output.clone().cpu().detach().numpy()
            return hook
        hook = make_hook(name, idx)
        hooks.append(layer.register_forward_hook(hook))

    # Process test data
    print("Evaluating test data")
    with torch.no_grad():
        _ = model(X_test)
        targets = y_test.cpu().numpy()

    # Remove hooks
    print("Removing hooks")
    for hook in hooks:
        hook.remove()

    # Process each layer and collect entropies and zero counts
    print("Building histograms")
    entropies = {}
    zero_counts = {}
    for layer_name, layer_data in layer_outputs.items():
        layer_entropies, layer_zero_counts = process_layer(
            layer_data, targets, layer_name, 
            args.output_dir, args.num_bins, 
            use_log_y=args.use_log_y,
            count_zeros=args.count_zeros
        )
        entropies[layer_name] = layer_entropies
        if args.count_zeros:
            zero_counts[layer_name] = layer_zero_counts

    # Save entropies to JSON
    entropy_path = os.path.join(args.output_dir, 'node_entropies.json')
    with open(entropy_path, 'w') as f:
        json.dump(entropies, f, indent=2)
        
    # Save zero counts to JSON if requested
    if args.count_zeros:
        zero_counts_path = os.path.join(args.output_dir, 'node_zero_counts.json')
        with open(zero_counts_path, 'w') as f:
            json.dump(zero_counts, f, indent=2)

if __name__ == '__main__':
    main()