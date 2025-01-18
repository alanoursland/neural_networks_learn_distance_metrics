import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from torchvision import datasets, transforms

def clear_gpu_memory():
    """Clear GPU memory and cache."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

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

def get_node_ranges(model: torch.nn.Module, X: torch.Tensor) -> Dict[str, Dict]:
    """Get activation ranges for each node in the perturbation layer."""
    ranges = {}
    activations = {}
    
    def hook_fn(name):
        def forward_hook(module, input, output):
            activations[name] = output.detach()
        return forward_hook
    
    # Register hook just for the perturbation layer
    hook = model.perturbation.register_forward_hook(hook_fn('perturbation'))
    
    # Forward pass to get activations
    with torch.no_grad():
        model(X)
    
    # Calculate ranges for each node
    activation = activations['perturbation']
    node_ranges = {
        'min': activation.min(dim=0).values.cpu().numpy().tolist(),
        'max': activation.max(dim=0).values.cpu().numpy().tolist(),
        'mean': activation.mean(dim=0).cpu().numpy().tolist(),
        'std': activation.std(dim=0).cpu().numpy().tolist()
    }
    ranges['perturbation'] = node_ranges
    
    # Remove hook
    hook.remove()
    
    return ranges

def evaluate_accuracy(outputs: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate accuracy from outputs."""
    return (torch.argmax(outputs, dim=1) == y).float().mean().item() * 100

def find_node_intersection(model: torch.nn.Module,
                         X: torch.Tensor,
                         y: torch.Tensor,
                         node_idx: int,
                         node_min: float,
                         node_max: float,
                         tolerance: float = 0.1) -> Dict:
    """
    Find intersection point of min/max perturbation accuracy curves using binary search.
    
    Args:
        model: The neural network model
        X: Input data tensor
        y: Target labels tensor
        node_idx: Index of node to analyze
        node_min: Minimum activation value for this node
        node_max: Maximum activation value for this node
        tolerance: Maximum allowed difference between accuracies to consider them equal
        
    Returns:
        Dictionary containing intersection details
    """
    def evaluate_base() -> float:
        # Get baseline accuracy
        model.perturbation.reset_perturbation()
        with torch.no_grad():
            base_acc = evaluate_accuracy(model(X), y)
        return base_acc
    
    def evaluate_threshold(threshold: float) -> Tuple[float, float]:
        """Test both perturbation types at given threshold."""
        # Test minimum perturbation (clip from below)
        model.perturbation.set_perturbation(min_val=threshold, max_val=float('inf'), node_idx=node_idx)
        with torch.no_grad():
            min_acc = evaluate_accuracy(model(X), y)
            
        # Test maximum perturbation (clip from above)
        model.perturbation.set_perturbation(min_val=float('-inf'), max_val=threshold, node_idx=node_idx)
        with torch.no_grad():
            max_acc = evaluate_accuracy(model(X), y)
            
        return min_acc, max_acc
    
    # Initialize search
    evaluations = 0
    left = node_min
    right = node_max
    base_acc = evaluate_base()
    # min_acc, _ = evaluate_threshold(node_max)
    # _, max_acc = evaluate_threshold(node_min)
    # if min_acc > base_acc or max_acc > base_acc:
    #     # This node's min-perturbation accuracy does not intersect the max-pertubation accuracy curve
    #     # min-perturbation increases accuracy and it does not decrease with maximal perturbation
    #     print(f"Node {node_idx} does not intersect. Setting intersection to {node_min}")
    #     print(f"{min_acc} {max_acc} {base_acc}")
    #     return {
    #         'threshold': float(node_min),
    #         'percentage': float(0.00),
    #         'accuracy': float(base_acc),
    #         'evaluations': 0,
    #         'min_acc': float(min_acc),
    #         'max_acc': float(max_acc)
    #     }
    
    # Binary search
    while True:
        mid = (left + right) / 2
        min_acc, max_acc = evaluate_threshold(mid)
        evaluations += 1

        # if min_acc > base_acc:
        #     print(f"Node {node_idx} improves at {mid}.")

        if evaluations > 100 or mid == node_max or mid == node_min:
            # Only ReLU2 shows these degenerate nodes.
            # Sometimes perturbations increase accuracy and in some of these cases there is no intersection between curves.
            print(f"Node {node_idx} does not intersect in {evaluations} steps. Setting intersection to {node_min}.")
            mid = node_min
            min_acc, max_acc = evaluate_threshold(mid)
            break

        # if node_idx == 43:
        #     print(f"{node_idx} {node_min:.2f} {node_max:.2f} {left:.2f} {mid:.2f} {right:.2f} {min_acc:.2f} {max_acc:.2f} {base_acc:.2f}")


        # Check if we've found intersection within tolerance
        # print(f"{evaluations} {left} {mid} {right} {min_acc} {max_acc}")
        if abs(min_acc - max_acc) <= tolerance:
            # Calculate percentage between min and max
            break
        
        # Update search bounds
        if min_acc > max_acc:
            left = mid  # Search upper half
        else:
            right = mid  # Search lower half

    percentage = (mid - node_min) / (node_max - node_min) * 100
    return {
        'threshold': float(mid),
        'percentage': float(percentage),
        'accuracy': float((min_acc + max_acc) / 2),
        'evaluations': evaluations,
        'min_acc': float(min_acc),
        'max_acc': float(max_acc)
    }


# def analyze_node_intersections(model: torch.nn.Module,
#                              X: torch.Tensor,
#                              y: torch.Tensor,
#                              node_ranges: Dict[str, Dict]) -> Dict:
#     """Find intersection points for all nodes in the perturbation layer."""
#     ranges = node_ranges['perturbation']
#     num_nodes = len(ranges['min'])
#     intersections = []
    
#     for node_idx in range(num_nodes):
#         node_min = ranges['min'][node_idx]
#         node_max = ranges['max'][node_idx]
        
#         # Find intersection
#         intersection = find_node_intersection(
#             model, X, y, node_idx, node_min, node_max
#         )
#         intersections.append(intersection)
        
#         # Print progress
#         if node_idx % 10 == 0:
#             print(f"Found intersection for node {node_idx}/{num_nodes}")
    
#     return {
#         'node_intersections': intersections,
#         'summary': {
#             'mean_percentage': float(np.mean([x['percentage'] for x in intersections])),
#             'std_percentage': float(np.std([x['percentage'] for x in intersections])),
#             'mean_evaluations': float(np.mean([x['evaluations'] for x in intersections])),
#             'total_evaluations': sum(x['evaluations'] for x in intersections),
#             'mean_accuracy': float(np.mean([x['accuracy'] for x in intersections])),
#             'std_accuracy': float(np.std([x['accuracy'] for x in intersections]))
#         }
#     }

def analyze_node_perturbations(model: torch.nn.Module, 
                             X: torch.Tensor, 
                             y: torch.Tensor,
                             node_ranges: Dict[str, Dict]) -> Dict:
    """Analyze perturbation effects for each node."""
    results = {}
    ranges = node_ranges['perturbation']
    num_nodes = len(ranges['min'])
    
    # Initialize results structure
    layer_results = {
        'min_acc': [],
        'max_acc': []
    }
    
    # Get baseline accuracy
    with torch.no_grad():
        base_acc = evaluate_accuracy(model(X), y)
    
    # Test each node
    for node_idx in range(num_nodes):
        node_min = ranges['min'][node_idx]
        node_max = ranges['max'][node_idx]
        
        # print(f"\t\t{node_idx} perturbation accuracy")
        # Test minimum perturbation
        model.perturbation.set_perturbation(
            min_val=node_max,  # Set to maximum (100% min perturbation)
            max_val=float('inf'),
            node_idx=node_idx
        )
        with torch.no_grad():
            min_acc = evaluate_accuracy(model(X), y)
        
        # Test maximum perturbation
        model.perturbation.set_perturbation(
            min_val=float('-inf'),
            max_val=node_min,  # Set to minimum (100% max perturbation)
            node_idx=node_idx
        )
        with torch.no_grad():
            max_acc = evaluate_accuracy(model(X), y)
        
        # Store accuracy
        layer_results['min_acc'].append(float(base_acc - min_acc))
        layer_results['max_acc'].append(float(base_acc - max_acc))
        
        # print(f"\t\t{node_idx} find_node_intersection")
        # Find intersection
        intersection = find_node_intersection(
            model, X, y, node_idx, node_min, node_max
        )
        if 'intersections' not in layer_results:
            layer_results['intersections'] = []
        layer_results['intersections'].append(intersection)
        
        # Reset perturbation
        model.perturbation.reset_perturbation()
        
        # Print progress
        if node_idx % 10 == 0:
            print(f"Processed node {node_idx}/{num_nodes}")
    
    results['perturbation'] = layer_results
    return results
    
def analyze_accuracy_distribution(results: Dict) -> Dict:
    """Analyze the distribution of perturbation accuracies."""
    stats = {}
    
    layer_results = results['perturbation']
    min_accs = np.array(layer_results['min_acc'])
    max_accs = np.array(layer_results['max_acc'])
    intersections = layer_results['intersections']
    
    stats['perturbation'] = {
        'min_acc': {
            'mean': float(np.mean(min_accs)),
            'std': float(np.std(min_accs)),
            'median': float(np.median(min_accs)),
            'quartiles': [float(x) for x in np.percentile(min_accs, [25, 75])],
            'max': float(np.max(min_accs)),
            'min': float(np.min(min_accs))
        },
        'max_acc': {
            'mean': float(np.mean(max_accs)),
            'std': float(np.std(max_accs)),
            'median': float(np.median(max_accs)),
            'quartiles': [float(x) for x in np.percentile(max_accs, [25, 75])],
            'max': float(np.max(max_accs)),
            'min': float(np.min(max_accs))
        },
        'accuracy_ratio': {
            'mean': float(np.mean(min_accs / np.maximum(max_accs, 1e-10))),
            'std': float(np.std(min_accs / np.maximum(max_accs, 1e-10))),
            'median': float(np.median(min_accs / np.maximum(max_accs, 1e-10)))
        },
        'intersections': {
            'mean_percentage': float(np.mean([x['percentage'] for x in intersections])),
            'std_percentage': float(np.std([x['percentage'] for x in intersections])),
            'mean_evaluations': float(np.mean([x['evaluations'] for x in intersections])),
            'total_evaluations': sum(x['evaluations'] for x in intersections),
            'mean_accuracy': float(np.mean([x['accuracy'] for x in intersections])),
            'std_accuracy': float(np.std([x['accuracy'] for x in intersections]))
        }
    }
    
    return stats
    
def plot_accuracy_distributions(results: Dict, output_dir: Path):
    """Create visualizations of accuracy distributions."""
    plt.style.use('seaborn-v0_8-darkgrid')
    layer_results = results['perturbation']
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    data = [
        layer_results['min_acc'],
        layer_results['max_acc']
    ]
    
    sns.violinplot(data=data)
    plt.xticks([0, 1], ['Min Perturbation', 'Max Perturbation'])
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Distribution for Perturbation Layer')
    plt.savefig(output_dir / 'perturbation_accuracy_distribution.png')
    plt.close()
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(layer_results['min_acc'], 
               layer_results['max_acc'], 
               alpha=0.5)
    plt.xlabel('Min Perturbation Accuracy (%)')
    plt.ylabel('Max Perturbation Accuracy (%)')
    plt.title('Accuracy Correlation for Perturbation Layer')
    
    # Add diagonal line
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    plt.savefig(output_dir / 'perturbation_accuracy_correlation.png')
    plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path('results/node_perturbation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, y_train = load_data(device)
    
    # Process each model
    model_dirs = list(Path('results/models').glob("*"))
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        # print(f"\nProcessing {model_name}")
        
        # Create model directory
        model_output_dir = results_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_run_results = []
        for run_dir in model_dir.glob("[0-9]*"):
            run_id = run_dir.name
            print(f"\nProcessing {model_name} {run_id}")
            
            model_path = run_dir / f"{model_name}.pt"
            if not model_path.exists():
                continue
            
            # Load model
            model = torch.load(model_path)
            model.to(device)
            model.eval()
            
            # Get node ranges
            # print("\tget_node_ranges")
            node_ranges = get_node_ranges(model, X_train)
            
            # Analyze perturbations
            # print("\tanalyze_node_perturbations")
            results = analyze_node_perturbations(model, X_train, y_train, node_ranges)
            
            # Analyze distributions
            # print("\tanalyze_accuracy_distribution")
            stats = analyze_accuracy_distribution(results)
            
            # Store results
            run_results = {
                'node_ranges': node_ranges,
                'perturbation_results': results,
                'accuracy_stats': stats
            }
            
            # Save run results
            with open(model_output_dir / f'run_{run_id}_results.json', 'w') as f:
                json.dump(run_results, f, indent=2)
            
            # Create visualizations
            plot_accuracy_distributions(results, model_output_dir)
            
            all_run_results.append(stats)
            
            # Clear memory
            model.cpu()
            del model
            clear_gpu_memory()
        
        # Generate summary statistics across runs
        summary_stats = {
            'perturbation': {
                'min_acc': {
                    'mean_of_means': np.mean([run['perturbation']['min_acc']['mean'] 
                                            for run in all_run_results]),
                    'std_of_means': np.std([run['perturbation']['min_acc']['mean'] 
                                          for run in all_run_results])
                },
                'max_acc': {
                    'mean_of_means': np.mean([run['perturbation']['max_acc']['mean'] 
                                            for run in all_run_results]),
                    'std_of_means': np.std([run['perturbation']['max_acc']['mean'] 
                                          for run in all_run_results])
                },
                'accuracy_ratio': {
                    'mean_of_means': np.mean([run['perturbation']['accuracy_ratio']['mean'] 
                                            for run in all_run_results]),
                    'std_of_means': np.std([run['perturbation']['accuracy_ratio']['mean'] 
                                          for run in all_run_results])
                }
            }   
        }
        
        # Save summary statistics
        with open(model_output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)

        # # Create histogram of equilibrium values
        # plt.figure(figsize=(10, 6))
        # plt.hist([run['perturbation']['intersections']['mean_percentage'] 
        #           for run in all_run_results], bins=20)
        # plt.xlabel('Equilibrium Value (%)')
        # plt.ylabel('Frequency')
        # plt.title(f'Histogram of Equilibrium Values for {model_name}')
        # plt.savefig(model_output_dir / f'equilibrium_values_histogram.png')
        # plt.close()

if __name__ == "__main__":
    main()