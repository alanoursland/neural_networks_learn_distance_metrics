import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from typing import Dict, List, Tuple
    
def load_data(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST training data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    X_train = train_dataset.data.float().to(device).reshape(-1, 28*28)
    y_train = train_dataset.targets.to(device)
    
    return X_train, y_train

def compute_statistics(points: torch.Tensor) -> Dict[str, float]:
    """Compute statistics for equilibrium points using GPU operations."""
    sorted_points, _ = torch.sort(points)
    n_points = points.size(0)
    
    # Calculate quartile indices
    q1_idx = int(n_points * 0.25)
    q3_idx = int(n_points * 0.75)
    
    return {
        'mean': float(torch.mean(points).item()),
        'std': float(torch.std(points).item()),
        'median': float(sorted_points[n_points // 2].item()),
        'q1': float(sorted_points[q1_idx].item()),
        'q3': float(sorted_points[q3_idx].item()),
        'min': float(torch.min(points).item()),
        'max': float(torch.max(points).item()),
        'n_points': n_points
    }

def analyze_equilibrium_points(results_dir: Path, output_dir: Path, device: torch.device):
    """Analyze equilibrium points across all models and runs using GPU operations."""
    model_points = {}
    model_stats = {}
    
    # Process each model directory
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        points_list = []
        
        # Process each run's results
        for result_file in model_dir.glob('run_*_results.json'):
            with open(result_file, 'r') as f:
                run_data = json.load(f)
            
            # Extract intersection percentages and convert to tensor
            points = [node_data['percentage'] 
                     for node_data in run_data['perturbation_results']['perturbation']['intersections']]
            points_list.extend(points)
        
        # Convert to tensor and move to GPU
        model_points[model_name] = torch.tensor(points_list, device=device)
        
        # Calculate statistics
        model_stats[model_name] = compute_statistics(model_points[model_name])
    
    # Save statistics
    with open(output_dir / 'equilibrium_stats.json', 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    # Create histograms
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, points) in enumerate(model_points.items()):
        ax = axes[idx]
        # Move to CPU only for plotting
        sns.histplot(points.cpu().numpy(), bins=50, ax=ax)
        ax.set_title(f'{model_name}\nμ={model_stats[model_name]["mean"]:.1f}%, σ={model_stats[model_name]["std"]:.1f}%')
        ax.set_xlabel('Equilibrium Point (%)')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equilibrium_distributions.png')
    plt.close()
    
    # Create box plot comparison
    plt.figure(figsize=(12, 6))
    data = [points.cpu().numpy() for points in model_points.values()]
    plt.boxplot(data, labels=list(model_points.keys()))
    plt.xticks(rotation=45)
    plt.title('Equilibrium Point Distribution Comparison')
    plt.ylabel('Equilibrium Point (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'equilibrium_boxplot.png')
    plt.close()
    
    return model_stats, model_points

def get_node_activations(model: torch.nn.Module, 
                        X: torch.Tensor) -> torch.Tensor:
    """Capture activations at the perturbation layer."""
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    # Register hook for perturbation layer
    hook = model.perturbation.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(X)
    
    # Remove hook
    hook.remove()
    
    return activations

def analyze_class_distributions(model: torch.nn.Module,
                              X: torch.Tensor,
                              y: torch.Tensor,
                              equilibrium_points: torch.Tensor) -> List[torch.Tensor]:
    """Analyze class distributions above/below equilibrium points for each node."""
    # Get activations for all inputs
    activations = get_node_activations(model, X)
    num_nodes = activations.size(1)
    num_classes = 10
    
    # Move equilibrium points to GPU if not already there
    if not isinstance(equilibrium_points, torch.Tensor):
        equilibrium_points = torch.tensor(equilibrium_points, device=X.device)
    
    # Initialize storage for all nodes at once
    node_distributions = torch.zeros(num_nodes, num_classes, 2, device=X.device)
    
    # Process all nodes in parallel where possible
    for class_idx in range(num_classes):
        class_mask = (y == class_idx)
        class_activations = activations[class_mask]
        
        # Compare against thresholds (broadcasting)
        below_mask = class_activations <= equilibrium_points.view(1, -1)
        above_mask = ~below_mask
        
        # Count occurrences (sum across batch dimension)
        node_distributions[:, class_idx, 0] = below_mask.sum(dim=0).float()
        node_distributions[:, class_idx, 1] = above_mask.sum(dim=0).float()
    
    # Normalize to percentages (along last dimension)
    node_distributions = node_distributions / node_distributions.sum(dim=-1, keepdim=True)
    
    # Round to 2 decimal places
    node_distributions = torch.round(node_distributions * 100) / 100
    
    # Convert to list of tensors for compatibility with existing code
    return [dist for dist in node_distributions]

def save_distributions(distributions: List[torch.Tensor],
                      model_name: str,
                      run_id: str,
                      output_dir: Path):
    """Save class distributions to JSON file with transposed format."""
    dist_data = []
    for node_idx, dist in enumerate(distributions):
        # Transpose the distributions from [10, 2] to [2, 10]
        transposed_dist = dist.t().cpu().tolist()
        node_data = {
            'node_idx': node_idx,
            'distributions': transposed_dist  # Now will be 2 arrays of 10 elements each
        }
        dist_data.append(node_data)
    
    output_file = output_dir / f'{model_name}_run_{run_id}_distributions.json'
    with open(output_file, 'w') as f:
        json.dump(dist_data, f, indent=2)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)  # Disable gradient tracking for all operations
    
    # Setup directories
    results_dir = Path('results/node_perturbation')
    analysis_dir = Path('results/node_features')
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    equilibrium_dir = analysis_dir / 'equilibrium'
    distribution_dir = analysis_dir / 'distributions'
    equilibrium_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze equilibrium points
    print("Analyzing equilibrium points...")
    model_stats, model_points = analyze_equilibrium_points(results_dir, equilibrium_dir, device)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data(device)
    
    # Process each model for class distributions
    print("Analyzing class distributions...")
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        print(f"\nProcessing {model_name}")
        
        # Process each run
        for result_file in model_dir.glob('run_*_results.json'):
            run_id = result_file.stem.split('_')[1]
            print(f"  Processing run {run_id}")
            
            # Load model
            model_path = Path('results/models') / model_name / run_id / f"{model_name}.pt"
            if not model_path.exists():
                continue
            
            model = torch.load(model_path)
            model.to(device)
            model.eval()
            
            # Load equilibrium points
            with open(result_file, 'r') as f:
                run_data = json.load(f)
            equilibrium_points = torch.tensor([
                intersection['threshold'] 
                for intersection in run_data['perturbation_results']['perturbation']['intersections']
            ], device=device)
            
            # Analyze distributions
            distributions = analyze_class_distributions(model, X_train, y_train, equilibrium_points)
            
            # Save results
            save_distributions(distributions, model_name, run_id, distribution_dir)
            
            # Clear GPU memory
            model.cpu()
            del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()