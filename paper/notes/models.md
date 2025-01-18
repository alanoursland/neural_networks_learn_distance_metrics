# Model Descriptions

This document outlines the six models implemented in `models.py` and describes what each model is expected to learn, along with the reasoning behind these expectations.

---

## 1. **Model_ReLU**
- **Description**: A two-layer MLP with a ReLU activation function.
- **Expected Behavior**: A standard model using ReLU. The linear layer can learn both intensity and distance views by negating its input weights.
- **Purpose**: Serves as a control for the other experiments.

---

## 2. **Model_Abs**
- **Description**: A two-layer MLP with an absolute value (Abs) activation function.
- **Expected Behavior**: A standard model using Abs. The linear layer can learn both intensity and distance views by negating its input weights.
- **Purpose**: Serves as a control for the other experiments.

---

## 3. **Model_ReLU2**
- **Description**: A two-layer MLP with an additional ReLU on the output layer.
- **Expected Behavior**: 
  - Due to the CrossEntropyLoss criterion, this model is forced to learn an **Intensity View**.
  - It must do so using positive values and cannot swap representations by negating the input weights.
- **Purpose**: Tests the ability of linear-ReLU configurations to learn intensity views.

---

## 4. **Model_Abs2**
- **Description**: A two-layer MLP with an additional Abs activation on the output layer.
- **Expected Behavior**:
  - Due to the CrossEntropyLoss criterion, this model is forced to learn an **Intensity View**.
  - It must do so using positive values and cannot swap representations by negating the input weights.
- **Purpose**: Tests the ability of linear-Abs configurations to learn intensity views.

---

## 5. **Model_ReLU2_Neg**
- **Description**: A two-layer MLP with an additional ReLU on the output layer, followed by a negated output.
- **Expected Behavior**:
  - This model must output a **Negative Intensity View**, which forces the linear-ReLU configuration to learn a **Positive Distance View**.
- **Purpose**: Tests the ability of linear-ReLU configurations to learn distance views.

---

## 6. **Model_Abs2_Neg**
- **Description**: A two-layer MLP with an additional Abs activation on the output layer, followed by a negated output.
- **Expected Behavior**:
  - This model must output a **Negative Intensity View**, which forces the linear-Abs configuration to learn a **Positive Distance View**.
- **Purpose**: Tests the ability of linear-Abs configurations to learn distance views.

---

# Summary

The six models explore the effects of activation functions (ReLU, Abs) and transformations (Neg) on the network's ability to balance distance- and intensity-based representations. By systematically comparing these models, we aim to understand how neural networks adapt their learned representations under different constraints and activation behaviors.

---

### Note on Bias
The second linear layer in each model does not have a bias. This prevents the conversion of \( y = (-W)x + b \) back into a positive representation, preserving the negated output. We tested each model against its version with a bias and found that over 20 runs, differences were not statistically significant.
