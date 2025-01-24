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

\section{Background 2}

\subsection{The Evolution of Neural Network Representations}
Neural networks have evolved significantly since McCulloch and Pitts introduced their binary threshold model in 1943. Their pioneering definition $\text{TRUE} = (Wx > b)$ established an intensity-based interpretation of neural activations that persists today. The field's progression through perceptrons \cite{rosenblatt1958} to modern deep learning architectures has largely maintained this view, where larger activation values indicate stronger feature presence.

However, recent theoretical work suggests neural networks might naturally compute statistical distance metrics \cite{oursland2024}, particularly the Mahalanobis distance. This perspective introduces a fundamental tension: while final classification layers must use intensity-based representations due to CrossEntropyLoss, hidden layers might prefer distance-based encodings where smaller activations indicate stronger feature matches.

\subsection{Distance versus Intensity Representations}
This tension manifests in two competing interpretations of neural activations:

\subsubsection{Distance-Based Features}
In distance-based representations, activations measure how far an input lies from a learned prototype or decision boundary. Smaller values indicate stronger matches, aligning with statistical metrics like Mahalanobis distance. This interpretation connects naturally with clustering algorithms and prototype learning.

\subsubsection{Intensity-Based Features}
The traditional intensity-based view interprets larger activations (whether positive or least negative) as indicating stronger feature presence. This aligns with the final classification objective, where the largest logit determines the predicted class under CrossEntropyLoss and LogSoftmax.

\subsection{Statistical Foundations}
The Mahalanobis distance provides a principled framework for understanding feature representations in neural networks. For a point $x$ with mean $\mu$ and covariance $\Sigma$, the distance is:

\[D_M(x) = \sqrt{(x - \mu)^T\Sigma^{-1}(x - \mu)}\]

This metric accounts for feature correlations and scaling, offering advantages over simpler measures like Euclidean distance. Through eigendecomposition of $\Sigma = V\Lambda V^T$, we can express this distance in terms of whitened coordinates:

\[D_M(x) = \|\Lambda^{-1/2}V^T(x - \mu)\|_2\]

This formulation reveals how linear layers with appropriate weights could compute components of the Mahalanobis distance, suggesting a natural connection between neural architectures and statistical distance metrics.

\subsection{Architectural Components}
Network architecture significantly influences representational capabilities:

\subsubsection{Activation Functions}
ReLU and Absolute Value activations impose different geometric constraints:
\begin{itemize}
    \item ReLU creates half-space decision boundaries, zeroing negative inputs
    \item Abs produces symmetric mappings, maintaining activation for all inputs
\end{itemize}

\subsubsection{Bias Terms}
The presence or absence of bias in linear layers critically affects representation:
\begin{itemize}
    \item With bias: Networks can artificially shift distance-based features to appear intensity-based
    \item Without bias: True representational preferences become apparent
\end{itemize}

\subsection{CrossEntropyLoss Framework}
The standard classification framework creates an inherent tension:
\begin{itemize}
    \item Final layer must use intensity-based encoding (largest logit wins)
    \item Hidden layers might naturally prefer distance-based representations
    \item Architecture must reconcile these competing pressures
\end{itemize}

This background motivates our investigation into how neural networks navigate the tension between distance and intensity representations under various architectural constraints.

\section{Background 3}

\subsection{Introduction to Representational Biases in Neural Networks}

The fundamental question driving this research is: \textit{how do neural networks encode information in their hidden layers?} This question has deep historical roots, tracing back to the binary threshold models of McCulloch and Pitts, the precursors to modern deep learning. \cite{1,32,33,34,35,36,37,38} Early models established the dominance of intensity-based interpretations, where larger activations signified stronger feature presence. However, contemporary theories have evolved to include distance metrics, where smaller activations indicate stronger matches to learned prototypes or decision boundaries.

This study delves into the implicit biases in feature representations, crucial for understanding network performance and design. \cite{1} Our core hypothesis is that hidden layers may inherently prefer distance representations, even when the final classification objective enforces an intensity-based view.

\subsection{Distance vs. Intensity Representations}

A key conceptual tension exists in neural networks between distance-based and intensity-based representations.

\textbf{Distance-Based Representations:} In this view, smaller activations imply stronger matches to learned prototypes, aligning with statistical metrics like Mahalanobis distance.

\textbf{Intensity-Based Representations:} This traditional perspective holds that larger activations signal a stronger presence of a feature, consistent with objectives like CrossEntropyLoss and LogSoftmax. \cite{2,14}

This study focuses on how architectural choices influence this representational tension within neural networks.

\subsection{Mathematical and Statistical Foundations}

We ground our analysis in the mathematical and statistical foundations of distance metrics and their geometric interpretations.

\textbf{Distance Metrics:} We define and formalize properties of key distance metrics, including Mahalanobis distance and Euclidean distance, emphasizing how they account for variance, correlation, and scaling in data.

\textbf{Geometric Interpretations:} We explore the geometric implications of hyperplanes and decision boundaries in classification, elucidating how bias terms shift these boundaries and influence feature representation.

\textbf{Whitening and Normalization:} We examine the statistical role of whitening and normalization in distance calculations, highlighting their impact on feature representation and alignment with network architecture.

\subsection{Network Components and Their Representational Impact}

We analyze how specific network components influence representational biases.

\textbf{Activation Functions:}

* \textbf{ReLU:} Its non-negative activations favor intensity-based interpretations but carry the risk of dead neurons. \cite{8,18}
* \textbf{Abs:} Its symmetric activations align with distance-based encodings, ensuring all nodes remain active. \cite{11,18}

\textbf{Geometric Effects:}

* \textbf{ReLU} creates half-space decision boundaries.
* \textbf{Abs} produces symmetric mappings around decision boundaries.

\textbf{Classification Framework:} CrossEntropyLoss and LogSoftmax enforce intensity-based outputs, where the largest logit determines the chosen class. This creates tension with intermediate layers that might encode features as distances. \cite{10,14}

\textbf{Bias Terms:} We stress the importance of bias removal in our experimental design.

* Preventing artificial shifts from distance-based to intensity-based representations. \cite{25}
* Clarifying the network's natural representational preferences.

\subsection{Architectural Constraints and Experimental Design}

Our experimental design employs architectural constraints to systematically probe representational biases.

\textbf{Experimental Architectures:} We utilize six two-layer architectures, each designed to test different aspects of representational biases. \cite{7,8,9}

* Variations in non-linearities (ReLU vs. Abs). \cite{11,12}
* Strategic use of Neg layers to force distance-based representations. \cite{19,20,21}
* Presence or absence of bias terms to isolate representational tendencies.

\textbf{Purpose of Constraints:}

* Isolating factors affecting distance-based vs. intensity-based representations. \cite{15}
* Revealing natural biases in how hidden layers encode information. \cite{32}

\subsection{Motivating Open Questions}

Our research is driven by several key open questions:

* Why do some architectures suffer catastrophic failure under intensity constraints? \cite{43,44}
* Can distance-based architectures match or exceed the performance of intensity-based ones?
* How do activation functions and geometric constraints interact to shape feature representations?
* What broader insights into neural network design can be gleaned from these findings? \cite{37}

By addressing these questions, we aim to contribute to a deeper understanding of neural networks and inform the development of more principled and interpretable architectures.