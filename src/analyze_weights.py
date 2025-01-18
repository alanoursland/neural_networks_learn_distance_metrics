import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import entropy
import os

def load_model_weights(model_path: Path):
    """Load weights from a saved model."""
    model = torch.load(model_path)
    # Extract weights from linear layers
    weights0 = model.linear0.weight.data.cpu().numpy().flatten()
    weights1 = model.linear1.weight.data.cpu().numpy().flatten()
    return weights0, weights1

def find_global_bounds(results_dir: Path):
    """Find global min/max weights across all models and runs."""
    global_min = float('inf')
    global_max = float('-inf')
    
    # Iterate through all model directories
    for model_dir in results_dir.glob("*"):
        if not model_dir.is_dir():
            continue
            
        # Process each run
        for run_dir in model_dir.glob("[0-9]*"):
            model_path = run_dir / f"{model_dir.name}.pt"
            if not model_path.exists():
                continue
                
            # Load weights and update bounds
            weights0, weights1 = load_model_weights(model_path)
            global_min = min(global_min, weights0.min(), weights1.min())
            global_max = max(global_max, weights0.max(), weights1.max())
    
    return global_min, global_max

def compute_histograms(weights, bins, range):
    """Compute histogram and normalize to get probability distribution."""
    hist, _ = np.histogram(weights, bins=bins, range=range)
    # Convert to probabilities
    prob_dist = hist / hist.sum()
    return prob_dist

def compute_kld(p, q):
    """Compute KL divergence between two distributions."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

def analyze_model_weights(results_dir: Path, output_dir: Path, n_bins=50):
    """Analyze weight distributions across all models and runs."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find global bounds
    global_min, global_max = find_global_bounds(results_dir)
    bins = np.linspace(global_min, global_max, n_bins + 1)
    
    # Dictionary to store all weights for computing overall distributions
    all_weights = {
        'layer0': {'model_type': [], 'weights': []},
        'layer1': {'model_type': [], 'weights': []}
    }
    
    # Process each model type
    model_stats = {}
    for model_dir in results_dir.glob("*"):
        if not model_dir.is_dir():
            continue
            
        model_type = model_dir.name
        model_stats[model_type] = {
            'layer0': {'runs': [], 'klds': [], 'pos_pct': [], 'neg_pct': []},
            'layer1': {'runs': [], 'klds': [], 'pos_pct': [], 'neg_pct': []}
        }
        
        # Process each run
        for run_dir in model_dir.glob("[0-9]*"):
            model_path = run_dir / f"{model_type}.pt"
            if not model_path.exists():
                continue
                
            weights0, weights1 = load_model_weights(model_path)
            
            # Store weights for overall distribution
            all_weights['layer0']['model_type'].extend([model_type] * len(weights0))
            all_weights['layer0']['weights'].extend(weights0)
            all_weights['layer1']['model_type'].extend([model_type] * len(weights1))
            all_weights['layer1']['weights'].extend(weights1)
            
            # Compute histograms for this run
            hist0 = compute_histograms(weights0, bins, (global_min, global_max))
            hist1 = compute_histograms(weights1, bins, (global_min, global_max))
            
            # Store histograms and compute positive/negative percentages
            model_stats[model_type]['layer0']['runs'].append(hist0)
            model_stats[model_type]['layer1']['runs'].append(hist1)
            
            model_stats[model_type]['layer0']['pos_pct'].append((weights0 > 0).mean() * 100)
            model_stats[model_type]['layer0']['neg_pct'].append((weights0 < 0).mean() * 100)
            model_stats[model_type]['layer1']['pos_pct'].append((weights1 > 0).mean() * 100)
            model_stats[model_type]['layer1']['neg_pct'].append((weights1 < 0).mean() * 100)
    
    # Compute overall distributions for each model type
    for model_type in model_stats.keys():
        # Convert runs to numpy array for easier computation
        model_stats[model_type]['layer0']['runs'] = np.array(model_stats[model_type]['layer0']['runs'])
        model_stats[model_type]['layer1']['runs'] = np.array(model_stats[model_type]['layer1']['runs'])
        
        # Compute mean distribution for this model type
        mean_dist0 = model_stats[model_type]['layer0']['runs'].mean(axis=0)
        mean_dist1 = model_stats[model_type]['layer1']['runs'].mean(axis=0)
        
        # Compute KLD for each run against mean distribution
        for run_idx in range(len(model_stats[model_type]['layer0']['runs'])):
            kld0 = compute_kld(model_stats[model_type]['layer0']['runs'][run_idx], mean_dist0)
            kld1 = compute_kld(model_stats[model_type]['layer1']['runs'][run_idx], mean_dist1)
            
            model_stats[model_type]['layer0']['klds'].append(kld0)
            model_stats[model_type]['layer1']['klds'].append(kld1)
    
    # Plot distributions
    plot_weight_distributions(model_stats, bins, output_dir)
    
    # Generate report
    generate_report(model_stats, output_dir)

def plot_weight_distributions(model_stats, bins, output_dir: Path):
    """Create plots of weight distributions."""
    plt.style.use('seaborn-v0_8-darkgrid')
    for layer in ['layer0', 'layer1']:
        plt.figure(figsize=(15, 10))
        
        for model_type, stats in model_stats.items():
            # Plot mean distribution with std deviation
            mean_dist = stats[layer]['runs'].mean(axis=0)
            std_dist = stats[layer]['runs'].std(axis=0)
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.plot(bin_centers, mean_dist, label=model_type, linewidth=2)
            plt.fill_between(bin_centers, 
                           mean_dist - std_dist, 
                           mean_dist + std_dist, 
                           alpha=0.2)
        
        plt.title(f'Weight Distribution - {layer}')
        plt.xlabel('Weight Value')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_dir / f'{layer}_distributions.png')
        plt.close()

def generate_report(model_stats, output_dir: Path):
    """Generate a report with weight distribution statistics."""
    report = ["Weight Distribution Analysis Report", "=" * 50, ""]
    
    for model_type, stats in model_stats.items():
        report.extend([
            f"\nModel: {model_type}",
            "-" * 30,
            "\nLayer 0:",
            f"Positive weights: {np.mean(stats['layer0']['pos_pct']):.2f}% ± {np.std(stats['layer0']['pos_pct']):.2f}%",
            f"Negative weights: {np.mean(stats['layer0']['neg_pct']):.2f}% ± {np.std(stats['layer0']['neg_pct']):.2f}%",
            f"Mean KLD: {np.mean(stats['layer0']['klds']):.4f} ± {np.std(stats['layer0']['klds']):.4f}",
            "\nLayer 1:",
            f"Positive weights: {np.mean(stats['layer1']['pos_pct']):.2f}% ± {np.std(stats['layer1']['pos_pct']):.2f}%",
            f"Negative weights: {np.mean(stats['layer1']['neg_pct']):.2f}% ± {np.std(stats['layer1']['neg_pct']):.2f}%",
            f"Mean KLD: {np.mean(stats['layer1']['klds']):.4f} ± {np.std(stats['layer1']['klds']):.4f}",
            ""
        ])
    
    with open(output_dir / 'weight_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))

if __name__ == "__main__":
    results_dir = Path("results/models")
    output_dir = Path("results/weight_dist")
    analyze_model_weights(results_dir, output_dir, n_bins=50)