Refined Outline (Final Version)
1. Introduction to Representational Biases in Neural Networks
Frame the fundamental question: How do neural networks encode information in hidden layers?
Historical Context:
From binary threshold models (McCulloch-Pitts) to continuous activations in modern deep learning.
How early models established the dominance of intensity-based interpretations.
Evolution of representation theories to include distance metrics.
Implicit biases in feature representations: Why they matter for performance and design.
Core hypothesis: Hidden layers may prefer distance representations, even under intensity-enforcing objectives.
2. Distance vs. Intensity Representations
Define the conceptual tension:
Distance-Based Representations:
Smaller activations = stronger matches (e.g., proximity to prototypes).
Connection to statistical metrics like Mahalanobis distance.
Intensity-Based Representations:
Larger activations = stronger feature presence.
Aligned with CrossEntropyLoss and LogSoftmax.
Highlight the study’s focus: Investigating how architectural choices influence this representational tension.
3. Mathematical and Statistical Foundations
Distance Metrics:
Definitions and formal properties (e.g., Mahalanobis distance, Euclidean distance).
How distance metrics account for variance, correlation, and scaling.
Geometric Interpretations:
Hyperplanes and decision boundaries in classification.
How bias terms shift decision boundaries and affect representation.
Whitening and Normalization:
Statistical role in distance calculations.
Impact on feature representation and alignment with network architecture.
4. Network Components and Their Representational Impact
Activation Functions:
ReLU: Non-negative activations favor intensity-based interpretations but risk dead neurons.
Abs: Symmetric activations align with distance-based encodings, maintaining active nodes.
Geometric effects:
ReLU creates half-space decision boundaries.
Abs produces symmetric mappings.
Classification Framework:
CrossEntropyLoss and LogSoftmax enforce intensity-based outputs (largest logit wins).
Tension with intermediate layers that might encode features as distances.
Bias Terms:
Importance of bias removal:
Prevents artificial shifts from distance to intensity.
Clarifies natural representational preferences.
5. Architectural Constraints and Experimental Design
Experimental Architectures:
Overview of six architectures, designed to probe representational biases:
Variations in non-linearities (ReLU vs. Abs).
Use of Neg layers to force distance representations.
Presence/absence of bias terms to isolate representational tendencies.
Purpose of Constraints:
How architectural components isolate factors affecting distance vs. intensity.
Why these configurations reveal natural biases in hidden layers.
6. Motivating Open Questions
Why do some architectures fail catastrophically under intensity constraints?
Can distance-based architectures match or outperform intensity-based ones?
How do activation functions and geometric constraints shape feature representations?
What broader insights into neural network design can be derived from these findings?
