# Dead Node Analysis Results

This analysis examines node activation patterns across different neural network architectures to understand how activation functions and transformations affect node utilization. The study focuses on dead nodes (never activate) and rarely active nodes (activate for <1% of inputs) across both training and test datasets.

## Key Findings

### Absolute Value Models (Abs, Abs2, Abs2-Neg)
All absolute value-based models show complete node utilization:
- No dead nodes in any layer
- No rarely active nodes in any layer
- Consistent behavior across all 20 training runs

This full utilization aligns with the distance metric interpretation, as absolute value activation allows both positive and negative directions to contribute meaningfully to the network's computations.

### ReLU-based Models

#### Base ReLU
Shows minimal node underutilization:
- First ReLU layer (activation0): 0.47% rarely active nodes
- No dead nodes in any layer
- Maintains good overall utilization

#### ReLU2
Exhibits severe node underutilization in its final activation layer:
- First ReLU layer (activation0): 0.12% rarely active nodes
- Final ReLU layer (activation1):
  - 22% dead nodes (mean: 2.20 ± 1.77 nodes)
  - 53.50% rarely active nodes (mean: 5.35 ± 1.53 nodes)
  - Total underutilization: 75.5% of output nodes

The high standard deviations indicate substantial variation between training runs, suggesting unstable learning dynamics. This severe underutilization explains the model's poor test accuracy (47.20%) and supports the hypothesis that forcing an intensity view through double ReLU activation significantly impairs the network's representational capacity.

#### ReLU2-Neg
Shows dramatically improved utilization compared to ReLU2:
- First ReLU layer: Only 0.12% rarely active nodes
- All subsequent layers: No dead or rarely active nodes
- Consistent utilization across runs

## Implications

1. **Distance vs. Intensity Views**
   - Absolute value models maintain full node utilization, consistent with effective distance-based representations
   - ReLU2's high rate of dead/rarely active nodes suggests problems with forced intensity views
   - Adding Neg transformation (ReLU2-Neg) recovers full utilization, indicating successful conversion to distance-based representation

2. **Architectural Considerations**
   - Double ReLU activation without Neg transformation severely constrains network capacity
   - Neg transformation appears to be crucial for maintaining node utilization in deeper ReLU networks
   - Absolute value activation naturally preserves full network capacity

3. **Training Stability**
   - Abs-based models show consistent behavior across runs
   - ReLU2 shows high variability in node utilization
   - ReLU2-Neg recovers training stability

## Conclusions

The dead node analysis provides strong empirical support for the paper's hypothesis about neural networks' preference for distance-based representations. The severe underutilization in ReLU2 and its recovery through the Neg transformation suggests that forcing intensity views can significantly impair network capacity, while enabling distance-based representations (either through Abs or ReLU-Neg) leads to full utilization of the network's computational resources.

This analysis also highlights the importance of analyzing node activation patterns for understanding neural network behavior, as the dramatic differences in node utilization help explain the observed performance differences between architectures.