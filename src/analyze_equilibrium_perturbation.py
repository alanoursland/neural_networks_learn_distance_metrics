import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_equilibrium_points():
    """Analyze equilibrium points across all models and runs."""
    results_dir = Path('results/node_perturbation')
    output_dir = Path('results/equilibrium_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all equilibrium points for each model
    model_points = {}
    model_stats = {}
    
    # Process each model directory
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        model_points[model_name] = []
        
        # Process each run's results
        for result_file in model_dir.glob('run_*_results.json'):
            with open(result_file, 'r') as f:
                run_data = json.load(f)
                
            # Extract intersection percentages for all nodes in this run
            intersections = run_data['perturbation_results']['perturbation']['intersections']
            percentages = [intersection['percentage'] for intersection in intersections]
            model_points[model_name].extend(percentages)
        
        # Calculate statistics for this model
        points = np.array(model_points[model_name])
        model_stats[model_name] = {
            'mean': float(np.mean(points)),
            'std': float(np.std(points)),
            'median': float(np.median(points)),
            'q1': float(np.percentile(points, 25)),
            'q3': float(np.percentile(points, 75)),
            'min': float(np.min(points)),
            'max': float(np.max(points)),
            'n_points': len(points)
        }
    
    # Save statistics
    with open(output_dir / 'equilibrium_stats.json', 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    # Create histograms
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, points) in enumerate(model_points.items()):
        ax = axes[idx]
        sns.histplot(points, bins=50, ax=ax)
        ax.set_title(f'{model_name}\nμ={model_stats[model_name]["mean"]:.1f}%, σ={model_stats[model_name]["std"]:.1f}%')
        ax.set_xlabel('Equilibrium Point (%)')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equilibrium_distributions.png')
    plt.close()
    
    # Create box plot comparison
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    for model_name, points in model_points.items():
        data.append(points)
        labels.append(model_name)
    
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=45)
    plt.title('Equilibrium Point Distribution Comparison')
    plt.ylabel('Equilibrium Point (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'equilibrium_boxplot.png')
    plt.close()

if __name__ == "__main__":
    analyze_equilibrium_points()