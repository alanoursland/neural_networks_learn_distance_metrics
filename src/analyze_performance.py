import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

def load_run_metrics(run_dir: Path) -> Dict:
    """Load metrics from a single run directory."""
    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_model_runs(model_dir: Path) -> List[Dict]:
    """Load all runs for a given model."""
    runs = []
    run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    for run_dir in run_dirs:
        metrics = load_run_metrics(run_dir)
        if metrics:
            runs.append(metrics)
    return runs

def analyze_model_performance(runs: List[Dict]) -> Dict:
    """Analyze performance statistics across runs."""
    final_test_accs = [run['test_acc'][-1] for run in runs]
    return {
        'mean': np.mean(final_test_accs),
        'std': np.std(final_test_accs),
        'min': np.min(final_test_accs),
        'max': np.max(final_test_accs),
        'median': np.median(final_test_accs),
        '95_ci': scipy.stats.t.interval(0.95, len(final_test_accs)-1, 
                                loc=np.mean(final_test_accs), 
                                scale=scipy.stats.sem(final_test_accs))
    }

def plot_learning_curves(results: Dict[str, List[Dict]], output_dir: Path):
    """Create separate learning curves for loss and error rates in log scale."""
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = sns.color_palette("colorblind", n_colors=20)
    
    # Define our four plots
    plot_configs = [
        {
            'metric': 'loss',
            'train_key': 'loss',
            'title': 'Training Loss',
            'filename': 'train_loss',
            'transform': np.log,
            'ylabel': 'Log(Loss)'
        },
        {
            'metric': 'accuracy',
            'train_key': 'train_acc',
            'title': 'Training Error Rate',
            'filename': 'train_error',
            'transform': lambda x: np.log(100 - x),
            'ylabel': 'Log(Error Rate)'
        },
        {
            'metric': 'accuracy',
            'train_key': 'test_acc',
            'title': 'Test Error Rate',
            'filename': 'test_error',
            'transform': lambda x: np.log(100 - x),
            'ylabel': 'Log(Error Rate)'
        }
    ]
    
    for config in plot_configs:
        plt.figure(figsize=(8, 6))
        
        for idx, (model_name, runs) in enumerate(results.items()):
            # Extract values for all runs
            try:
                curves = np.array([run[config['train_key']] for run in runs])
                
                # Apply transformation (log)
                with np.errstate(divide='ignore', invalid='ignore'):
                    curves = config['transform'](curves)
                
                # Remove any infinite values that might come from log(0)
                curves = np.nan_to_num(curves, neginf=np.nanmin(curves[curves > -np.inf]))
                
                # Calculate mean and std
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                
                # Create x-axis (epochs)
                epochs = range(1, len(mean_curve) + 1)
                
                # Plot mean line
                plt.plot(epochs, mean_curve, label=model_name, color=colors[idx])
                
                # Plot standard deviation band
                plt.fill_between(epochs, 
                               mean_curve - std_curve, 
                               mean_curve + std_curve, 
                               alpha=0.2, 
                               color=colors[idx])
                
            except KeyError:
                print(f"Warning: {config['train_key']} not found for {model_name}")
                continue
        
        plt.xlabel('Epoch')
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f"{config['filename']}.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        
def plot_performance_distribution(results: Dict[str, Dict], output_dir: Path):
    """Create violin plots of final test accuracies across models."""
    data = []
    for model_name, model_stats in results.items():
        data.extend([(model_name, acc) for acc in model_stats['all_accuracies']])
    
    df = pd.DataFrame(data, columns=['Model', 'Test Accuracy'])
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Model', y='Test Accuracy')
    plt.xticks(rotation=45)
    plt.title('Distribution of Test Accuracies Across Models')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_distribution.png')
    plt.close()

def find_convergence_epoch(loss_values: List[float], window: int = 5, threshold: float = 0.0001) -> int:
    """
    Find the epoch where training converged.
    
    Args:
        loss_values: List of training loss values
        window: Number of epochs to look back for convergence check
        threshold: Maximum allowed change ratio for convergence
    
    Returns:
        Epoch number where convergence occurred (1-based indexing)
    """
    if len(loss_values) < window:
        return len(loss_values)
        
    for i in range(window, len(loss_values)):
        # Calculate relative changes over window
        changes = [(loss_values[i-j] - loss_values[i-j-1])/loss_values[i-j-1] 
                  for j in range(window)]
        
        # Check if all changes are below threshold
        if all(abs(change) < threshold for change in changes):
            return i
            
    return len(loss_values)  # If no convergence found, return last epoch

def generate_latex_table(results: Dict[str, Dict], runs: Dict[str, List[Dict]], output_dir: Path):
    """
    Generate a LaTeX table summarizing model performance.
    
    Args:
        results: Dictionary containing analysis results
        runs: Dictionary containing all runs for each model
        output_dir: Directory to save the LaTeX file
    """
    # Start table content
    table_content = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Test Acc.} & \textbf{Train Acc.} & \textbf{95\% CI} & \textbf{\#Params} & \textbf{Conv. Epoch} \\",
        r"& (\%) & (\%) & (\%) & (M) & \\",
        r"\midrule"
    ]
    
    for model_name, model_runs in runs.items():
        # Get test accuracy stats (already calculated)
        test_stats = results[model_name]
        
        # Calculate training accuracy stats
        final_train_accs = [run['train_acc'][-1] for run in model_runs]
        train_mean = np.mean(final_train_accs)
        train_std = np.std(final_train_accs)
        
        # Find average convergence epoch using training loss
        conv_epochs = []
        for run in model_runs:
            if 'train_loss' in run:
                conv_epochs.append(find_convergence_epoch(run['train_loss']))
        conv_epoch = int(np.mean(conv_epochs)) if conv_epochs else '-'
        
        # Get number of parameters if available
        num_params = model_runs[0].get('num_parameters', '-')
        if isinstance(num_params, (int, float)):
            num_params = f"{num_params/1e6:.1f}"  # Convert to millions
        
        # Format confidence interval
        ci_low, ci_high = test_stats['95_ci']
        
        # Add row to table
        row = (
            f"{model_name} & "
            f"{test_stats['mean']:.1f} ± {test_stats['std']:.1f} & "
            f"{train_mean:.1f} ± {train_std:.1f} & "
            f"[{ci_low:.1f}, {ci_high:.1f}] & "
            f"{num_params} & "
            f"{conv_epoch} \\\\"
        )
        table_content.append(row)
    
    # Complete the table
    table_content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Summary of model performance across multiple runs. Test and Train Acc. columns show mean ± standard deviation. The 95\% CI column shows confidence intervals for test accuracy. \#Params shows model size in millions of parameters. Conv. Epoch indicates the epoch where training loss converged (defined as change < 0.01\% over 5 epochs).}",
        r"\label{tab:model_results}",
        r"\end{table}"
    ])
    
    # Save to file
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write('\n'.join(table_content))

def main():
    # Configuration

    results_dir = Path('results')
    experiment_dir = results_dir / 'models_with_L2'  # Update this based on your experiment name
    output_dir =    results_dir / 'analysis_with_L2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    model_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

    # Load all runs for learning curves
    all_runs = {}
    for model_dir in model_dirs:
        runs = load_model_runs(model_dir)
        if runs:
            all_runs[model_dir.name] = runs
    
    
    # Analyze each model for summary statistics
    results = {}
    for model_dir in model_dirs:
        runs = load_model_runs(model_dir)
        if runs:
            final_accuracies = [run['test_acc'][-1] for run in runs]
            stats = analyze_model_performance(runs)
            stats['all_accuracies'] = final_accuracies
            results[model_dir.name] = stats
    
    # Generate summary report
    report = ["Model Performance Analysis\n"]
    report.append("=" * 50 + "\n")
    
    for model_name, stats in results.items():
        report.append(f"\n{model_name}:")
        report.append("-" * 30)
        report.append(f"Number of runs: {len(stats['all_accuracies'])}")
        report.append(f"Mean test accuracy: {stats['mean']:.2f}%")
        report.append(f"Standard deviation: {stats['std']:.2f}%")
        report.append(f"Min accuracy: {stats['min']:.2f}%")
        report.append(f"Max accuracy: {stats['max']:.2f}%")
        report.append(f"95% CI: [{stats['95_ci'][0]:.2f}%, {stats['95_ci'][1]:.2f}%]")
        report.append("")
    
    # Save report
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write('\n'.join(str(line) for line in report))
    
    # Generate learning curves
    plot_learning_curves(all_runs, output_dir)
    # Generate visualizations
    plot_performance_distribution(results, output_dir)
    
    # Generate statistical test report
    report = ["Statistical Analysis Betweeen Models\n"]
    report.append("=" * 50 + "\n")

    models = list(results.keys())
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = scipy.stats.ttest_ind(
                results[model1]['all_accuracies'],
                results[model2]['all_accuracies']
            )
            report.append(f"\n{model1} vs {model2}:")
            report.append(f"t-statistic: {t_stat:.4f}")
            report.append(f"p-value: {p_value:.4f}")
            report.append("")

    # Save report
    with open(output_dir / 'ttest_report.txt', 'w') as f:
        f.write('\n'.join(str(line) for line in report))
    
    generate_latex_table(results, all_runs, output_dir)

if __name__ == "__main__":
    main()