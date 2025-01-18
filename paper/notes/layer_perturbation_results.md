# Layer Perturbation Analysis Results

## Experimental Setup

The experiment uses a PerturbationLayer that allows controlled modification of activation values during testing while acting as an identity function during training. Two types of constraints were tested:

1. **Minimum Value Constraints**: Clipping all values below a threshold
2. **Maximum Value Constraints**: Clipping all values above a threshold

The thresholds were set as percentages between the minimum and maximum activation values observed in the layer. For example, a 35% minimum constraint means values are clipped to a point 35% of the way from the minimum to the maximum activation value.

## Key Findings

### Performance Dropoff Points (50% Accuracy Loss)

#### Minimum Value Constraints (Distance-like):
- Abs2-Neg: 38%
- Abs: 35%
- Abs2: 15%
- ReLU2-Neg: 11%
- ReLU: 9%
- ReLU2: 8%

#### Maximum Value Constraints (Intensity-like):
- Abs2-Neg: 35%
- Abs: 30%
- Abs2: 8%
- ReLU2-Neg: 1%
- ReLU: 1%
- ReLU2: 1%

### Universal Bias Pattern

A crucial observation is that all models show stronger sensitivity to minimum value constraints compared to maximum value constraints. This bias appears to be a fundamental property rather than an artifact of specific architectures:

1. **ReLU Family**:
   - Shows extreme bias (9% vs 1% for base ReLU)
   - Pattern persists even in poorly performing ReLU2
   - ReLU2-Neg improves overall performance but maintains the bias pattern

2. **Abs Family**:
   - Shows more balanced behavior but still biased (35% vs 30% for base Abs)
   - Bias is significantly less pronounced than in ReLU models
   - Pattern consistent across all Abs variants

## Model-Specific Analysis

### Base Models (ReLU and Abs):
- ReLU shows high sensitivity to minimum constraints (96.47% → 55.37% with just 5% perturbation)
- ReLU maintains better performance under maximum constraints (gradual decline from 98.44% to 81.72%)
- Abs shows remarkable stability up to 20% perturbation in both directions
- Abs maintains 99.99% accuracy with small perturbations

### ReLU2 (Double ReLU):
- Severely degraded baseline performance (~47%)
- Extremely sensitive to minimum constraints
- Maximum constraints show minimal impact but cannot improve performance
- High run-to-run variance (std dev up to 12.61%) suggesting unstable learning

### ReLU2-Neg (ReLU with Negation):
- Recovers strong baseline performance (~97%)
- Shows similar sensitivity pattern to base ReLU
- More robust than ReLU2 but maintains sensitivity to minimum constraints
- Better stability between runs than ReLU2

### Abs2 and Abs2-Neg:
- Abs2 maintains high performance (~99%)
- Shows gradual degradation under minimum constraints
- Very resilient to maximum constraints
- Abs2-Neg performs worse than Abs2, contrary to initial theoretical predictions

## Theoretical Implications

These results suggest a refinement of our understanding of how neural networks process information:

1. **Fundamental Distance Metric**:
   - The universal bias toward minimum value sensitivity suggests all these networks might be fundamentally operating on distance-like metrics
   - The difference between ReLU and Abs may be in how severely they constrain this natural tendency
   - Previous interpretations of Abs vs ReLU behavior may need revision

2. **Architectural Constraints**:
   - Rather than networks choosing between distance and intensity views, they might all operate on distance-like metrics but with varying degrees of constraint
   - This explains why forcing positive representations (ReLU2) is so detrimental - it severely restricts the network's ability to represent these naturally distance-based computations

3. **Role of Activation Functions**:
   - Activation functions might be better understood as constraints on how networks can represent distance-based computations
   - ReLU imposes stronger constraints, leading to more extreme bias
   - Abs allows more balanced representation while maintaining the fundamental distance-based nature

## Conclusion

These findings support but refine the thesis that neural networks naturally use distance metrics. Rather than being a property that some architectures have and others don't, it appears to be a fundamental characteristic that different architectures constrain to varying degrees. This suggests that optimal network design might focus on allowing networks to express these natural distance-based computations while maintaining necessary constraints for specific tasks.