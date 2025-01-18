

## 1. The Core Tension: Distance vs. Intensity

- **Distance-Based Feature Interpretation**  
  The theory (from prior work) suggests that hidden-layer activations in neural networks often behave like a distance metric, indicating “closeness” to a learned prototype or decision boundary. In such a paradigm, **smaller values** correspond to a stronger match (i.e., more “presence” of the feature).

- **Intensity-Based Feature Interpretation**  
  Standard training with **CrossEntropyLoss (CEL)** and **LogSoftmax** demands that the “winning” class have the *highest* logit (whether it is actually a large positive number or simply less negative than the others). This sets up an *intensity-based* perspective: **larger values** correspond to stronger feature presence. CrossEntropyLoss is applied the same across all models.

The **key tension** is that although CEL + LogSoftmax *interprets* logits in an intensity-based way (higher logit = more likely), neural networks may **still** internally prefer to encode data in a distance-like manner (smaller activation = closer match).

---

## 2. Six Architectures to Probe Representations

To test whether networks exhibit an *inherent bias* toward distance-like or intensity-like representations (or freely mix the two), the experiments use six architectures. They all share a similar backbone of two linear layers with standard ReLU and Abs nonlinear activations in between—but differ in how the final layers are constructed:

1. **x → Linear → ReLU → Linear → y**  
2. **x → Linear → Abs → Linear → y**  
3. **x → Linear → ReLU → Linear → ReLU → y**  
4. **x → Linear → ReLU → Linear → ReLU → Neg → y**  
5. **x → Linear → Abs → Linear → Abs → y**  
6. **x → Linear → Abs → Linear → Abs → Neg → y**  

### Non-Neg vs. Neg Variants
- **ReLU/Abs produce non-negative outputs** at their stage. Interpreted loosely, these can align with a “distance-like” notion (distances are non-negative).  
- **Neg** (i.e., \(y = -x\)) flips sign. If the penultimate layer is “distance-like” (small = strong match), Neg can turn that into negative values, which—relatively speaking—LogSoftmax can still interpret as a “high” logit if it is the *least* negative among classes.  
- **Without Neg**, the final logit might naturally remain non-negative, so the network could choose to keep an intensity framing or revert to a distance framing by adjusting weights.  
- The final output of each model must be an intensity representation. The Neg's before this force linear->activation to learn a distance representation.


### Simple Control Models vs. Strictly Constrained Models
- The **simpler control** models (variants 1 and 2) have no extra activation on the output. Here, the final linear layer can learn either distance-like or intensity-like by adjusting signs of its weights and adding biases.  
- The **“-Neg”** models (variants 4 and 6) explicitly force the network to produce *distance-based* values in the penultimate layer (non-negative via ReLU/Abs), which are then **inverted** to become negative intensities for the final logit.

Models 1 and 2 are controls. They are able to learn either distance or intensity representations because they can just learn negated weights.

Models 3 and 5 are forced to learn positive intensity representations.

Models 4 and 6 are forced to learn positive distance representations which are then converted to negative intensity representations.

---

## 3. Role of Bias Removal in the Final Layer

A central subtlety is **removing biases** in the second linear layer (and only the that layer). Normally, a bias term could *translate* activations: if the network internally prefers distance (i.e., smaller is “better”), it can just add a large positive bias \(b\) to shift those negative or small raw outputs into a positive range. This trivial shift can mask whether the network is actually using a distance-based or intensity-based representation. Bias provides enough flexibility to still learn its preferred representation as negative values and then translate it back up to a positive target representation.

By **removing** the bias in the final layer:

- The network cannot simply add a constant offset to turn a negative/distance representation into a positive/intensity representation.  
- It must either (a) use a **Neg** step (if provided) or (b) learn weights whose signs/values reflect the “pure” distance or intensity approach without relying on a bias shift.

This design aims to make it clearer whether the final logit emerges from a fundamentally distance-like or intensity-like internal representation.

---

## 4. Training Setup: Minimizing Confounds

All models are trained under **the same simplistic conditions**:
- **Dataset**: MNIST  
- **Optimizer**: Stochastic Gradient Descent (full-batch updates)  
- **Epochs**: 5000  
- **Fixed Learning Rate**  
- **20 Repetitions** per architecture

Because the goal is not raw performance but rather **to observe how the network develops representations**, the training setup is intentionally stripped down: no momentum, no adaptive methods, no partial mini-batches, and no early stopping. This simplicity **reduces confounds** and helps ensure that observed differences in how the models converge are due to **intrinsic biases** in representation learning, rather than fine-tuned hyperparameters.

Full-batch means the entire MNIST training set is used for each gradient step. The entire training set is moved to the GPU and evaluated on the model in a single operation.This is done for performance as well as simplifying hparams.

---

## 5. Putting It All Together

1. **Final Output → Always Interpreted by LogSoftmax**:  
   LogSoftmax can handle any real values (positive or negative). The “highest” logit—least negative or largest positive—wins. So even if a network’s penultimate layer yields non-negative (distance) values, **Neg** can flip those into negative logits, which can still be “largest” in a comparative sense.

2. **Distance vs. Intensity at Internal Layers**:  
   Despite the final output being used in an intensity-based classification scheme, earlier layers may be *fundamentally* encoding features in a distance-like manner (smaller = better match). The different architectures and the **Neg** step force or disallow certain sign manipulations, thus revealing whether the network “wants” to represent features as distances or intensities under various constraints.

3. **Bias Removal Exposes Underlying Preference**:  
   By removing bias, the network cannot trivially shift a negative/“distance-based” internal representation into a positive/logit-based intensity representation. It either must adapt in a more structural way (through negative weights or explicit Neg layers), or it must choose an intensity-based encoding from the get-go. This distinction is **non-trivial** and is precisely what the experiment aims to clarify.

---

## 6. Expected Outcome & Significance

While the actual **results** are not shared here, the **expectation** (based on prior theory) is that networks *might* exhibit a **bias** toward distance-like representations. The question is: **Do the constraints (ReLU/Abs + Neg + no bias) make that bias more evident, or do the networks adapt just as well to intensity-based features?**  

Regardless of whether the final logit is positive or negative, these experiments probe whether there is a *default or “natural” inclination* for neural networks to measure how *close* an input is to certain decision boundaries (distance) rather than purely how *large* a feature is (intensity). The cleanliness of the training regimen and the architectural variations are meant to **disentangle** these possibilities and provide empirical insight into representational biases.

---

### In Short
This experiment systematically manipulates final-layer activations (ReLU/Abs), sign flips (Neg), and bias terms to **force or prohibit** different paths from a potential “distance-based” internal encoding to the mandatory **intensity-based** output for classification. By doing so, it aims to reveal whether networks truly prefer to treat smaller values as “features present” (distance) or larger values as “features present” (intensity)—and how easily they can switch between the two.