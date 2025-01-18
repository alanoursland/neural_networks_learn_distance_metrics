import torch
import json
from pathlib import Path
from torchvision import datasets, transforms

def load_model(model_path: Path) -> torch.nn.Module:
    """Load a saved model and move to GPU."""
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def analyze_node_activations(model: torch.nn.Module, 
                           X: torch.Tensor,
                           device: torch.device) -> dict:
    """Analyze activation patterns of nodes in each layer using GPU operations."""
    activation_stats = {}
    hooks = []
    layer_activations = {}
    
    def hook_fn(name):
        def forward_hook(module, input, output):
            layer_activations[name] = output
        return forward_hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.ReLU)) or \
           any(x in str(type(module)) for x in ['Abs', 'Neg']):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Process entire dataset at once
    with torch.no_grad():
        model(X)
    
    total_samples = X.size(0)
    
    # Calculate statistics
    for name, activations in layer_activations.items():
        # Count nodes that are active (directly on GPU)
        active_counts = (torch.abs(activations) > 1e-6).sum(dim=0)
        activation_stats[name] = {
            'total_samples': total_samples,
            'dead_nodes': (active_counts == 0).sum().item(),
            'rarely_active_nodes': (active_counts < total_samples * 0.01).sum().item(),
            'total_nodes': activations.size(1)
        }
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats

def analyze_all_models(results_dir: Path, X_train: torch.Tensor, X_test: torch.Tensor, device: torch.device):
    """Analyze dead nodes for all models and runs on both training and test data."""
    output_dir = results_dir / 'dead_nodes'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dirs = [d for d in (results_dir / 'models').iterdir() if d.is_dir()]
    all_results = {}
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nAnalyzing {model_name}...")
        
        train_results = []
        test_results = []
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        
        for run_dir in run_dirs:
            model_path = run_dir / f"{model_name}.pt"
            print(f"Loading model {model_path}")
            
            if model_path.exists():
                model = load_model(model_path)
                if model is not None:
                    model.to(device)
                    
                    # Analyze on training data
                    train_stats = analyze_node_activations(model, X_train, device)
                    train_results.append(train_stats)
                    
                    # Analyze on test data
                    test_stats = analyze_node_activations(model, X_test, device)
                    test_results.append(test_stats)
                    
                    # Clear GPU memory
                    model.cpu()
                    del model
                    torch.cuda.empty_cache()
        
        if train_results and test_results:
            train_aggregated = aggregate_stats(train_results)
            test_aggregated = aggregate_stats(test_results)
            
            all_results[model_name] = {
                'train': train_aggregated,
                'test': test_aggregated
            }
    
    generate_reports(all_results, output_dir)

def aggregate_stats(results):
    """Aggregate statistics across runs."""
    aggregated_stats = {}
    num_runs = len(results)
    layer_names = results[0].keys()
    
    for layer in layer_names:
        dead_nodes = torch.tensor([run[layer]['dead_nodes'] for run in results])
        rarely_active = torch.tensor([run[layer]['rarely_active_nodes'] for run in results])
        total_nodes = results[0][layer]['total_nodes']
        
        aggregated_stats[layer] = {
            'avg_dead_nodes': float(dead_nodes.float().mean()),
            'std_dead_nodes': float(dead_nodes.float().std()),
            'avg_rarely_active': float(rarely_active.float().mean()),
            'std_rarely_active': float(rarely_active.float().std()),
            'total_nodes': total_nodes,
            'num_runs': num_runs
        }
    
    return aggregated_stats

def generate_reports(all_results, output_dir):
    """Generate summary reports for both training and test results."""
    for dataset_type in ['train', 'test']:
        report_lines = [f"Dead Node Analysis Report - {dataset_type.upper()}", "=" * 50, ""]
        
        for model_name, results in all_results.items():
            stats = results[dataset_type]
            report_lines.append(f"\nModel: {model_name}")
            report_lines.append("-" * 30)
            
            for layer_name, layer_stats in stats.items():
                if layer_name == "":  # Skip empty layer names
                    continue
                    
                report_lines.append(f"\nLayer: {layer_name}")
                report_lines.append(f"Total nodes: {layer_stats['total_nodes']}")
                report_lines.append(f"Number of runs analyzed: {layer_stats['num_runs']}")
                report_lines.append(f"Dead nodes (mean ± std): {layer_stats['avg_dead_nodes']:.2f} ± {layer_stats['std_dead_nodes']:.2f}")
                report_lines.append(f"Rarely active nodes (mean ± std): {layer_stats['avg_rarely_active']:.2f} ± {layer_stats['std_rarely_active']:.2f}")
                report_lines.append(f"Dead node percentage: {(layer_stats['avg_dead_nodes']/layer_stats['total_nodes']*100):.2f}%")
                report_lines.append(f"Rarely active percentage: {(layer_stats['avg_rarely_active']/layer_stats['total_nodes']*100):.2f}%")
        
        # Save reports
        with open(output_dir / f'summary_report_{dataset_type}.txt', 'w') as f:
            f.write('\n'.join(report_lines))
    
    # Save detailed results
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("No GPU found. Exiting...")
        return

    # Load MNIST dataset directly to GPU
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Move entire datasets to GPU
    X_train = train_dataset.data.float().to(device).reshape(-1, 28*28)
    X_test = test_dataset.data.float().to(device).reshape(-1, 28*28)
    
    # Run analysis
    results_dir = Path('results')
    analyze_all_models(results_dir, X_train, X_test, device)

if __name__ == "__main__":
    main()