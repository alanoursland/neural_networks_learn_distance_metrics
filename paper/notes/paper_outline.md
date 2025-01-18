
## **1. Introduction**
- **High-Level Overview**:
  - Neural networks are traditionally interpreted through an intensity-based lens, where higher activations signify stronger feature presence. However, recent theoretical and empirical findings suggest that networks may naturally prefer distance-based representations, where smaller activations correspond to proximity to learned features.
- **Core Problem**:
  - Loss functions like CrossEntropyLoss enforce intensity-based interpretations, creating a potential conflict with the internal preferences of neural networks for distance-based representations.
- **Research Gap**:
  - Limited empirical evidence for how architectural constraints, such as bias removal and Neg transformations, influence representational preferences.
- **Our Approach**:
  - Systematic experiments using six neural network architectures to isolate the effects of activation functions, transformations, and bias removal.
  - Focus on exposing intrinsic representational preferences under controlled conditions.
- **Key Findings**:
  - ReLU2 fails catastrophically (~47%), while ReLU2-Neg recovers (~95%) by leveraging distance representations.
  - Abs models demonstrate unexpected robustness, suggesting unique alignment with distance metrics.
  - Bias removal plays a critical role in revealing intrinsic preferences.
- **Impact**:
  - Challenges the conventional intensity-based interpretation of neural networks.
  - Provides actionable insights for architecture design and empirical validation of distance-metric theories.

---

## **2. Prior Work**
- **Historical Development of Activation Interpretations**:
  - **Early Work**:
    - McCulloch-Pitts (1943): Binary thresholding introduced the idea of activations representing feature presence.
    - Rosenblatt (1958): The perceptron model expanded this to continuous activation values, reinforcing the intensity-based view.
  - **Modern Deep Learning**:
    - Introduction of ReLU (Nair & Hinton, 2010) and its variants (Leaky ReLU, PReLU, ELU) cemented intensity-based interpretations.
- **Distance Metrics in Machine Learning**:
  - Central role of distance metrics like Euclidean and Mahalanobis distance in clustering and classification.
  - Applications in algorithms like k-means, Gaussian Mixture Models, and RBF networks.
  - Siamese and metric-learning networks explicitly use distance metrics but are rarely connected to standard neural network architectures.
- **Neural Network Interpretability**:
  - Feature visualization and saliency map methods primarily focus on intensity-based interpretations.
  - Explainable AI frameworks (e.g., SHAP, LIME) enhance transparency but do not address internal representational preferences.
- **Adversarial Robustness and Perturbation Sensitivity**:
  - Studies reveal high sensitivity to small perturbations, often near decision boundaries, aligning with a distance-metric perspective.
  - Foundational work by Goodfellow et al. (2014) and Szegedy et al. (2013) highlights the role of decision boundaries in network behavior.
- **Gaps and Motivations**:
  - Limited studies explore whether neural networks inherently prefer distance representations.
  - Need for experiments that isolate the effects of activation functions, transformations, and bias removal.

---

## **3. Background**
- **Theoretical Framework**:
  - **Statistical Distance Metrics**:
    - Mahalanobis distance accounts for covariance and is widely used in clustering.
    - Neural network activations have been shown to approximate such metrics.
  - **Linear Layers and Distances**:
    - Mathematical connections between weights, biases, and distance computation.
    - Bias terms enable trivial translations between representations, complicating interpretability.
    - Non-uniqueness of whitening transformations introduces flexibility in learned representations.
- **Core Tension**:
  - Loss functions like CrossEntropyLoss demand intensity-based logits, while networks may internally favor distance representations.
  - Sign flips (e.g., Neg) and bias removal expose this tension by forcing or preventing certain translations.
- **Architectural Components**:
  - **Activation Functions**:
    - ReLU zeros out negatives, creating sparse outputs aligned with intensity metrics.
    - Abs folds negatives into positives, preserving magnitude and aligning with distance metrics.
  - **Neg Transformation**:
    - Converts distance representations to intensity metrics by flipping signs.
  - **Bias Removal**:
    - Prevents trivial translations between distance and intensity, exposing the network’s true representational preference.

---

## **4. Experimental Design**
- **Objective**:
  - Investigate representational preferences by systematically manipulating architectural constraints.
- **Model Architectures**:
  - **Control Models**:
    - ReLU and Abs serve as baselines, capable of learning both representations.
  - **Double Activation Models**:
    - ReLU2 and Abs2 enforce positive intensity representations by stacking activations.
  - **Neg Transformation Models**:
    - ReLU2-Neg and Abs2-Neg enforce distance representations that are converted to intensity metrics.
- **Key Constraints**:
  - **Bias Removal**:
    - Ensures that representational preferences are exposed rather than masked by trivial translations.
  - **Activation Functions**:
    - Shape representation space and interact with bias removal in unique ways.
- **Training Protocol**:
  - **Dataset**: MNIST.
  - **Optimization**: Full-batch SGD, 5000 epochs, consistent learning rate.
  - **Statistical Robustness**: 20 runs per model to ensure significance.
- **Evaluation Metrics**:
  - Accuracy, variance, dead node counts, and statistical significance testing.

---

## **5. Results**
- **Baseline Performance**:
  - ReLU: 95.64% ± 0.19%.
  - Abs: 95.26% ± 0.19%.
  - High stability and robustness across runs.
- **ReLU Variants**:
  - ReLU2: Catastrophic failure (47.20% ± 12.00%).
  - ReLU2-Neg: Near-complete recovery (94.93% ± 0.15%).
- **Abs Variants**:
  - Abs2: Robust performance (95.35% ± 0.17%).
  - Abs2-Neg: Performance drop (90.08% ± 2.56%).
- **Statistical Comparisons**:
  - Significant differences across models highlight the role of constraints.
  - Dead node analysis reveals mechanisms behind ReLU2’s failure.

---

## **6. Discussion**
- **Analysis of Key Findings**:
  - ReLU2 fails due to dead nodes and representational constraints, while ReLU2-Neg thrives by converting distance to intensity.
  - Abs models’ robustness suggests inherent alignment with distance metrics.
  - Abs2-Neg’s drop indicates challenges in achieving separation without bias.
- **Theoretical Implications**:
  - Supports the hypothesis that networks prefer distance representations under certain conditions.
  - Highlights the unique behaviors of activation functions in shaping learned representations.
- **Practical Insights**:
  - Recommendations for architecture design:
    - Use Neg transformations to align representations with task requirements.
    - Carefully consider the role of biases in optimizing performance.
- **Limitations and Future Work**:
  - Generalization to complex datasets and architectures.
  - Exploring optimization dynamics and deeper networks.

---

## **7. Conclusion**
- **Summary of Contributions**:
  - Empirical validation of distance-metric theories in neural networks.
  - Insights into the role of architectural constraints on representational preferences.
  - Practical recommendations for network design.
- **Broader Implications**:
  - Challenges conventional intensity-based interpretations.
  - Opens new avenues for interpretability and architectural innovation.
- **Future Directions**:
  - Scaling experiments to more complex datasets.
  - Investigating interactions between optimization and representational preferences.
  - Exploring applications in robust and interpretable network design.
