# Performance Results Summary

## Baseline Performance
- **ReLU**: 95.64% ± 0.19%
- **Abs**: 95.26% ± 0.19%

Both baseline models achieved strong performance with minimal variance between runs, suggesting they can effectively learn the required representations.

## Theory and Predictions
Our theory posited that neural networks naturally learn distance-based representations rather than intensity-based ones. We predicted that:
1. Forcing models into intensity views (via double positive activation) should degrade performance
2. Enabling distance views through negation should improve or restore performance
3. This pattern should hold for both ReLU and Abs activation functions

## ReLU Results - Supporting the Theory
- **ReLU2**: 47.20% ± 12.00%
  - Catastrophic performance degradation
  - High variability between runs
  - 22% dead nodes in final activation layer
  - Supports theory that forcing intensity view impairs learning

- **ReLU2-Neg**: 94.93% ± 0.15%
  - Nearly complete recovery of performance
  - Stable performance across runs
  - No dead nodes
  - Supports theory that enabling distance view restores learning capability

## Abs Results - Contradicting the Theory
- **Abs2**: 95.35% ± 0.17%
  - Maintains baseline performance
  - Stable across runs
  - No dead nodes
  - Contradicts theory by performing well with double positive activation

- **Abs2-Neg**: 90.08% ± 2.56%
  - Significant performance drop
  - High variability between runs
  - Contradicts theory by performing worse when supposedly enabled to learn distance views

## Key Insights
1. The ReLU results strongly support our theoretical framework, with clear evidence that forcing intensity views impairs learning while enabling distance views restores it.

2. The Abs results suggest there are fundamental differences in how ReLU and Abs handle representations that aren't captured by our current theoretical framework.

3. The presence of dead nodes in ReLU2 provides a clear mechanism for its failure, while the absence of dead nodes in Abs2 suggests it can learn effective representations despite the double positive activation constraint.

These results indicate that while the distance-metric theory may correctly describe aspects of neural network learning, the relationship between activation functions and distance/intensity views is more complex than initially theorized.