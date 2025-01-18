import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json
import gc

def clear_gpu_memory():
    """Clear GPU memory and cache."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(f'cuda:{i}'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def load_data(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST training data."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    X_train = train_dataset.data.float().to(device).reshape(-1, 28*28)
    y_train = train_dataset.targets.to(device)
    
    return X_train, y_train

def evaluate_accuracy(outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy directly from outputs."""
    return (torch.argmax(outputs, dim=1) == y).float().mean() * 100

def get_activation_ranges(model: torch.nn.Module, X: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Calculate activation ranges for each layer."""
    ranges = {}
    activations = {}
    
    def hook_fn(name):
        def forward_hook(module, input, output):
            activations[name] = output.detach()
        return forward_hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU)) or \
           any(x in str(type(module)) for x in ['Abs']):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        model(X)
    
    # Calculate ranges
    for name, activation in activations.items():
        ranges[name] = {
            'min': float(activation.min()),
            'max': float(activation.max()),
            'mean': float(activation.mean()),
            'std': float(activation.std())
        }
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return ranges

def test_perturbations(model: torch.nn.Module, 
                      X: torch.Tensor, 
                      y: torch.Tensor,
                      n_steps: int) -> Dict[str, torch.Tensor]:
    """Test model under various perturbation conditions with batched processing."""
    device = X.device
    results = {
        'min': torch.zeros(n_steps, device=device),
        'max': torch.zeros(n_steps, device=device)
    }
    
    with torch.no_grad():
        # Get baseline accuracy
        outputs = model(X)
        baseline_acc = evaluate_accuracy(outputs, y)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        
        # Get activation ranges
        ranges = get_activation_ranges(model, X)
        layer_min = min(r['min'] for r in ranges.values())
        layer_max = max(r['max'] for r in ranges.values())
        
        # Create percentage steps tensor and perturbation values
        percentages = torch.linspace(0, 100, n_steps, device=device)
        vals = layer_min + (layer_max - layer_min) * (percentages / 100)
        
        # Process in batches of perturbations
        batch_size = 10  # Number of perturbation values to process at once
        for start_idx in range(0, n_steps, batch_size):
            end_idx = min(start_idx + batch_size, n_steps)
            batch_vals = vals[start_idx:end_idx]
            
            # Test minimum value perturbations
            for i, val in enumerate(batch_vals):
                model.perturbation.set_perturbation(min_val=val, max_val=torch.inf)
                outputs = model(X)
                results['min'][start_idx + i] = evaluate_accuracy(outputs, y)
            
            # Test maximum value perturbations
            for i, val in enumerate(batch_vals):
                model.perturbation.set_perturbation(min_val=-torch.inf, max_val=val)
                outputs = model(X)
                results['max'][start_idx + i] = evaluate_accuracy(outputs, y)
            
            # Print progress less frequently
            if start_idx % 30 == 0:
                print(f"Progress: {start_idx/n_steps*100:.0f}%")
        
        # Reset perturbation
        model.perturbation.reset_perturbation()
    
    return results

def plot_results(results: Dict[str, Dict[str, torch.Tensor]], 
                n_steps: int,
                output_dir: Path):
    """Create visualization of perturbation results."""
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = sns.color_palette("colorblind", n_colors=len(results))
    percentages = torch.linspace(0, 100, n_steps).cpu()
    
    # Plot each perturbation type separately
    for perturb_type in ['min', 'max']:
        plt.figure(figsize=(12, 8))
        
        for idx, (model_name, model_results) in enumerate(results.items()):
            accuracies = model_results[perturb_type].cpu()
            if accuracies.dim() > 1:
                mean_acc = accuracies.mean(dim=0)
                std_acc = accuracies.std(dim=0)
            else:
                mean_acc = accuracies
                std_acc = torch.zeros_like(accuracies)
            
            plt.plot(percentages, mean_acc, label=model_name, color=colors[idx])
            plt.fill_between(percentages, 
                           mean_acc - std_acc,
                           mean_acc + std_acc,
                           alpha=0.2,
                           color=colors[idx])
        
        plt.xlabel('Perturbation Percentage')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{perturb_type.capitalize()} Value Perturbation Effects')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f'sensitivity_{perturb_type}.png')
        plt.close()
    
    # Create combined plot
    plt.figure(figsize=(15, 10))
    line_styles = ['-', '--']
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        for p_idx, perturb_type in enumerate(['min', 'max']):
            accuracies = model_results[perturb_type].cpu()
            mean_acc = accuracies.mean(dim=0) if accuracies.dim() > 1 else accuracies
            
            plt.plot(percentages, 
                    mean_acc, 
                    label=f'{model_name} ({perturb_type})',
                    color=colors[idx],
                    linestyle=line_styles[p_idx])
    
    plt.xlabel('Perturbation Percentage')
    plt.ylabel('Accuracy (%)')
    plt.title('Combined Perturbation Effects')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'sensitivity_combined.png')
    plt.close()

def analyze_layer_perturbation():
    """Main function to run layer perturbation analysis."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path('results/layer_perturbation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    X_train, y_train = load_data(device)
    
    # Define number of steps
    n_steps = 101  # 0 to 100 in steps of 1
    
    # Process each model type
    results = {}
    model_dirs = list(Path('results/models').glob("*"))
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"\nProcessing {model_name}")
        
        # Initialize storage tensors for this model's results
        model_results = {
            'min': torch.zeros(20, n_steps, device=device),  # 20 runs
            'max': torch.zeros(20, n_steps, device=device)
        }
        
        run_count = 0
        for run_dir in model_dir.glob("[0-9]*"):
            print(f"  Run {model_dir} {run_dir.name}")
            model_path = run_dir / f"{model_name}.pt"
            
            if not model_path.exists():
                continue
                
            # Load model
            model = torch.load(model_path)
            model.to(device)
            model.eval()
            
            # Test perturbations
            run_results = test_perturbations(model, X_train, y_train, n_steps)
            
            # Store results
            for key in model_results:
                model_results[key][run_count] = run_results[key]
            
            run_count += 1
            
            # Clear memory
            model.cpu()
            del model
            clear_gpu_memory()
        
        # Trim unused rows if some runs failed
        if run_count < 20:
            for key in model_results:
                model_results[key] = model_results[key][:run_count]
        
        results[model_name] = model_results
    
    # Save results
    with open(results_dir / 'perturbation_results.json', 'w') as f:
        json_results = {
            model: {
                key: val.cpu().tolist() for key, val in model_results.items()
            }
            for model, model_results in results.items()
        }
        json.dump(json_results, f)
    
    # Generate plots
    plot_results(results, n_steps, results_dir)
    
    # Generate summary report
    report_lines = ["Layer Perturbation Analysis Results", "=" * 50, ""]
    
    for model_name, model_results in results.items():
        report_lines.extend([
            f"\nModel: {model_name}",
            "-" * 30
        ])
        
        for perturb_type in ['min', 'max']:
            accuracies = model_results[perturb_type]
            mean_acc = accuracies.mean(dim=0)
            std_acc = accuracies.std(dim=0)
            
            report_lines.extend([
                f"\n{perturb_type.capitalize()} Perturbation:",
                f"  Mean accuracy range: {mean_acc.min():.2f}% - {mean_acc.max():.2f}%",
                f"  Std dev range: {std_acc.min():.2f}% - {std_acc.max():.2f}%",
                f"   1% perturbation accuracy: {mean_acc[1]:.2f}% ± {std_acc[1]:.2f}%",
                f"   5% perturbation accuracy: {mean_acc[5]:.2f}% ± {std_acc[5]:.2f}%",
                f"  10% perturbation accuracy: {mean_acc[10]:.2f}% ± {std_acc[10]:.2f}%",
                f"  20% perturbation accuracy: {mean_acc[20]:.2f}% ± {std_acc[20]:.2f}%",
                f"  30% perturbation accuracy: {mean_acc[30]:.2f}% ± {std_acc[30]:.2f}%",
                f"  40% perturbation accuracy: {mean_acc[40]:.2f}% ± {std_acc[40]:.2f}%",
                f"  50% perturbation accuracy: {mean_acc[50]:.2f}% ± {std_acc[50]:.2f}%",
                f"  60% perturbation accuracy: {mean_acc[60]:.2f}% ± {std_acc[60]:.2f}%",
                f"  70% perturbation accuracy: {mean_acc[70]:.2f}% ± {std_acc[70]:.2f}%",
                f"  80% perturbation accuracy: {mean_acc[80]:.2f}% ± {std_acc[80]:.2f}%",
                f"  90% perturbation accuracy: {mean_acc[90]:.2f}% ± {std_acc[90]:.2f}%"
            ])
    
    with open(results_dir / 'layer_perturbation_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    analyze_layer_perturbation()