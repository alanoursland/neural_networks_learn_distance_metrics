# Project Overview

## **Title**: Exploring Distance and Intensity Views in Neural Networks

---

## **Introduction**
This project investigates how neural networks learn representations through the lens of **distance-based** and **intensity-based** views. Building upon theoretical and empirical findings from two prior research efforts, the project explores how activation functions (ReLU, Abs) and transformations (Neg) interact with standard training criteria like CrossEntropyLoss to influence learned representations. The goal is to understand whether neural networks naturally prefer distance-based views and how they adapt under constraints requiring intensity views.

---

## **Background**
1. **Prior Research**:
   - **Theoretical Foundation**: Neural network activations can approximate statistical distance metrics, such as the Mahalanobis distance, which align with a **Distance View**.
   - **Empirical Validation**: Perturbation studies demonstrated that neural networks often rely on features near decision boundaries (distance metrics) rather than relying on activation magnitude (intensity metrics).

2. **Core Problem**:
   - Standard loss functions like CrossEntropyLoss require outputs in an **Intensity View**.
   - Distance-based representations must be converted into intensity-based representations (or vice versa) to meet these requirements.
   - Activation functions like ReLU and Abs, which constrain outputs to non-negative values, may influence the network's ability to learn and convert between these views.

---

## **Goals**
1. **Investigate View Preferences**:
   - Determine whether neural networks naturally align with distance-based representations under various constraints.
   - Explore the ability of networks to adapt to intensity-based representations when required by the loss function.

2. **Activation Function Behavior**:
   - Evaluate how ReLU and Abs activations impact the network's ability to learn distance and intensity views.
   - Examine the role of stacked activations (e.g., ReLU2, Abs2) in reinforcing view preferences.

3. **Transformation Effects**:
   - Assess how the inclusion of a Neg transformation influences representation learning, especially when converting between views.

4. **Control and Baseline Comparisons**:
   - Compare models with standard activations to those with added transformations to understand the interplay between architecture, activation, and loss criteria.

---

## **Methodology**
1. **Dataset**:
   - Experiments are conducted on the MNIST dataset, a standard benchmark for digit recognition tasks.

2. **Models**:
   - Six models are implemented to systematically explore the effects of activation functions and transformations on learned representations:
     - `Model_ReLU`
     - `Model_Abs`
     - `Model_ReLU2`
     - `Model_Abs2`
     - `Model_ReLU2_Neg`
     - `Model_Abs2_Neg`
   - The second linear layer in all models is **bias-free** to prevent trivial conversions between views.

3. **Training**:
   - Models are trained using CrossEntropyLoss, which enforces an intensity-based output.
   - Multiple runs are conducted for statistical robustness, with results averaged to evaluate trends.

4. **Evaluation Metrics**:
   - Training and test accuracy.
   - Convergence speed (epochs to reach a threshold accuracy).
   - Sensitivity to distance and intensity perturbations.

---

## **Key Hypotheses**
1. **Control Models**:
   - Models with single activations (`Model_ReLU`, `Model_Abs`) can learn both views and serve as controls.

2. **ReLU Variants**:
   - ReLU-based models (`Model_ReLU2`, `Model_ReLU2_Neg`) are expected to favor intensity views due to ReLU's zeroing behavior but can adapt to distance views when Neg is introduced.

3. **Abs Variants**:
   - Abs-based models (`Model_Abs2`, `Model_Abs2_Neg`) are more aligned with distance views due to the folding effect of Abs but must adapt to intensity requirements when combined with CrossEntropyLoss.

4. **Stacking Effects**:
   - Adding stacked activations (`ReLU2`, `Abs2`) reinforces the view associated with the activation function, making adaptation harder but potentially enhancing representation robustness.

---

## **Expected Contributions**
1. **Insight into Representation Learning**:
   - Clearer understanding of how neural networks navigate distance and intensity representations.
   - Evidence on whether ReLU and Abs activations favor one view over the other.

2. **Practical Implications**:
   - Guidance on activation function selection based on task requirements.
   - Potential for more interpretable neural network architectures by aligning learned representations with theoretical insights.

3. **Framework for Future Research**:
   - Establishes a foundation for studying the interaction of activation functions, loss criteria, and transformations in neural networks.

---

## **Conclusion**
This project advances the understanding of neural network representation learning by systematically exploring the balance between distance- and intensity-based views. The findings will contribute to both theoretical research and practical model design, offering new tools and insights for neural network interpretability and performance optimization.
