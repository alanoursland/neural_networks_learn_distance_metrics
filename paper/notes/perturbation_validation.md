# Validating Model Sensitivity Through Perturbation Analysis

## Overview

To validate that our models exhibit similar sensitivity patterns to those in "Neural Networks Use Distance Metrics", we need to conduct a systematic perturbation analysis that maps to the prior paper's methodology while using our new perturbation layer implementation.

## Mapping Between Papers

### Original Paper's Approach
- Used scale and offset parameters to modify decision boundaries
- Measured accuracy impact of boundary shifts
- Demonstrated sensitivity to small distance-based perturbations
- Showed robustness to large intensity-based perturbations

### Current Implementation Requirements
- Use value clamping instead of boundary shifts
- Maintain equivalent test coverage
- Produce comparable sensitivity measurements
- Generate similar visualization outputs

## Testing Methodology

### Test Parameters
- Sample 100 points (1% intervals from 0% to 100%)
- Test both min and max perturbations at each point
- Average results across all 20 training runs
- Compare sensitivity patterns between models

### For Each Model

1. **Baseline Measurement**
   ```python
   def get_baseline(model, test_loader):
       model.perturbation.reset_perturbation()
       return evaluate_accuracy(model, test_loader)
   ```

2. **Value Range Analysis**
   ```python
   def analyze_value_ranges(model, test_loader):
       # Get activation ranges for normalization
       ranges = {}
       for name, layer in model.named_modules():
           if isinstance(layer, nn.ReLU) or isinstance(layer, Abs):
               ranges[name] = get_activation_range(model, layer, test_loader)
       return ranges
   ```

3. **Perturbation Testing**
   ```python
   def test_perturbations(model, test_loader, ranges):
       results = {
           'min_perturb': [],
           'max_perturb': []
       }
       
       for percentage in range(0, 101):
           # Convert percentage to relative value
           rel_val = percentage / 100.0
           
           # Test minimum value perturbation
           min_acc = test_min_perturbation(model, test_loader, rel_val, ranges)
           results['min_perturb'].append(min_acc)
           
           # Test maximum value perturbation
           max_acc = test_max_perturbation(model, test_loader, rel_val, ranges)
           results['max_perturb'].append(max_acc)
           
       return results
   ```

### Specific Tests

1. **Minimum Value Perturbation**
   ```python
   def test_min_perturbation(model, test_loader, rel_val, ranges):
       accuracies = []
       for layer_name, range_info in ranges.items():
           min_val = range_info['min'] * rel_val
           model.perturbation.set_perturbation(min_val=min_val)
           acc = evaluate_accuracy(model, test_loader)
           accuracies.append(acc)
           model.perturbation.reset_perturbation()
       return np.mean(accuracies)
   ```

2. **Maximum Value Perturbation**
   ```python
   def test_max_perturbation(model, test_loader, rel_val, ranges):
       accuracies = []
       for layer_name, range_info in ranges.items():
           max_val = range_info['max'] * rel_val
           model.perturbation.set_perturbation(max_val=max_val)
           acc = evaluate_accuracy(model, test_loader)
           accuracies.append(acc)
           model.perturbation.reset_perturbation()
       return np.mean(accuracies)
   ```

## Expected Results Analysis

### Visualizations

1. **Sensitivity Curves**
   ```python
   def plot_sensitivity_curves(results):
       plt.figure(figsize=(12, 6))
       for model_name, model_results in results.items():
           plt.plot(range(101), model_results['min_perturb'], 
                   label=f'{model_name} (min)', linestyle='--')
           plt.plot(range(101), model_results['max_perturb'], 
                   label=f'{model_name} (max)', linestyle='-')
       plt.xlabel('Perturbation Percentage')
       plt.ylabel('Accuracy')
       plt.title('Model Sensitivity to Value Perturbations')
       plt.legend()
       plt.grid(True)
       plt.savefig('sensitivity_curves.png')
   ```

2. **Comparative Analysis**
   - Create heatmaps showing sensitivity differences between models
   - Generate statistical significance tests for sensitivity patterns
   - Compare patterns to original paper's findings

### Validation Criteria

1. **Distance Metric Patterns**
   - High sensitivity to small value perturbations
   - Clear accuracy degradation with small changes
   - Similar patterns between ReLU and Abs models

2. **Intensity Metric Patterns**
   - Lower sensitivity to large value perturbations
   - Maintained accuracy under scaling
   - Differences between ReLU and Abs models

## Implementation Notes

1. **Optimization**
   - Cache activation ranges to avoid recomputation
   - Parallelize perturbation tests where possible
   - Use vectorized operations for efficiency

2. **Robustness**
   - Handle numerical stability issues
   - Implement error checking for range calculations
   - Log anomalous results for investigation

3. **Data Collection**
   - Save detailed results for each test
   - Record timing information
   - Track memory usage during testing

## Success Criteria

1. **Pattern Matching**
   - Sensitivity curves should match original paper
   - Statistical significance of differences should be low
   - Relative model behaviors should align

2. **Metric Validation**
   - Distance metric sensitivity should be clear
   - Intensity metric robustness should be evident
   - Model-specific patterns should emerge

3. **Documentation**
   - Clear mapping between methodologies
   - Detailed explanation of differences
   - Analysis of any discrepancies