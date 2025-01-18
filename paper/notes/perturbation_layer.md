# Perturbation Layer Documentation

## Overview
The PerturbationLayer is a custom PyTorch module designed to modify activation values during model analysis. Unlike the perturbation analysis in the paper which modified decision boundaries through scale and offset parameters, this implementation directly clamps activation values to specified ranges.

## Implementation Details

### Key Components
```python
class PerturbationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_perturbation()
```

The layer maintains three key parameters:
- `min_val`: Minimum allowed activation value (default: -inf)
- `max_val`: Maximum allowed activation value (default: inf)
- `node_idx`: Optional index for perturbing specific nodes (default: None)

### Behavior Modes

1. **Identity Mode (Default)**
   - When `min_val = -inf` and `max_val = inf`
   - Layer passes values through unchanged
   - Used during normal training

2. **Global Perturbation Mode**
   - When `node_idx = None`
   - Clamps all activations to `[min_val, max_val]`
   - Used for analyzing entire layer responses

3. **Single Node Perturbation Mode**
   - When `node_idx` is specified
   - Only perturbs the specified node
   - Useful for analyzing individual node contributions

### Key Methods

1. `set_perturbation(min_val, max_val, node_idx)`
   - Configures perturbation parameters
   - Can target all nodes or a specific node
   - Sets value clamping range

2. `reset_perturbation()`
   - Returns layer to identity mode
   - Resets all parameters to defaults

3. `get_perturbation()`
   - Returns current perturbation settings
   - Useful for logging and analysis

## Usage in Research

### Distance vs. Intensity Testing
- Can simulate distance metric behavior by clamping values close to decision boundaries
- Can test intensity metric behavior by allowing wide value ranges
- Enables controlled comparison of both representation types

### Node Significance Analysis
- Individual node perturbation allows measuring each node's contribution
- Can identify critical nodes vs. redundant ones
- Helps understand network's learned representations

## Key Differences from Paper Implementation

1. **Direct Value Control**
   - Paper: Modified decision boundaries through scale/offset
   - This: Directly clamps activation values

2. **Granularity**
   - Paper: Applied perturbations to entire layers
   - This: Can target individual nodes

3. **Training Impact**
   - Paper: Perturbations affected training
   - This: Acts as identity during training, only used for analysis

4. **Flexibility**
   - Paper: Fixed perturbation types
   - This: Configurable perturbation ranges and targets

## Advantages

1. **Precision**: Enables fine-grained control over activation values
2. **Isolation**: Can study individual nodes without affecting others
3. **Reversibility**: Easy to toggle between perturbed and normal operation
4. **Analysis**: Better suited for post-training analysis
5. **Integration**: Seamlessly fits into PyTorch model architecture