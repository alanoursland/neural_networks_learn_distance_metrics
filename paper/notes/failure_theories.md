
1. Dead Outputs and Disjunctive Intensity in ReLU2
ReLU2’s Catastrophic Failure
You note that ReLU2 often ends up with many “dead” output units—neurons or features that saturate to zero and no longer receive meaningful gradient updates. Because we’ve removed biases and forced a double ReLU, the model effectively has no straightforward way to “shift” those dead features back into an active range once they go negative.

Disjunctive Intensity Metric
You propose that the network compensates by learning a disjunctive intensity scheme: it tries to push non-target features to “absent” (or near-zero) while somehow keeping the target class’s representation active. Put differently:

“Distance features” for digits 
≠

= target get placed close to the decision boundary (small margin, on the verge of being clipped by ReLU).
Once a feature is accidentally pushed onto the negative side (pre-activation < 0), the ReLU’s zero cutoff kills it—leading to a “dead” feature.
Without bias or any sign-flipping mechanism at the output, the network can’t easily revive those dead features or offset them.
The result is a compromised model that fails to separate classes effectively, as evidenced by the ~50% accuracy.

2. Conjunctive Distance Metric in ReLU2-Neg
Why Neg Solves the Problem
By adding a Neg layer at the end, ReLU2-Neg can represent “the target” in a more direct, conjunctive way:

The network pushes the target features to be close to the boundary (small non-negative numbers).
The Neg flips these small numbers into the least negative final logits—which, under CrossEntropyLoss, become the “highest” in a relative sense.
Non-target features that get large ReLU outputs become very negative logits after Neg, safely putting them in the background.
Fewer Dead Nodes
Because the model only has to get one digit’s features near zero (the target) rather than nine, it doesn’t risk saturating so many features at zero. Fewer outputs wind up “dead,” and the model maintains enough expressivity to hit ~95% accuracy.

3. Abs2-Neg and the Role of Bias
Abs2-Neg Hitting 90%
You suggest that Abs2-Neg in particular might require biases to fully work, or at least strongly benefit from them. With two Abs layers and a Neg at the end, the network is heavily constrained:

Abs
Abs forces all intermediate features to be non-negative (similar to ReLU but without the zero-saturation for negative inputs—though negative pre-activations become positive after Abs).
Neg
Neg then flips them at the end.
No bias means the final shift to “make the target class stand out” can’t be handled by simply adding a constant offset.
Shifting the Target to Zero
You hypothesize that if we reinstated biases, Abs2-Neg might place the target class activation at zero (a true “distance is zero” scenario) while pushing all other classes slightly above zero before Neg. The bias term could then shift those other classes further from zero, creating more separation and yielding higher accuracy.

Put differently, the network might want to say, “The target is at distance = 0” and “Everything else is some positive distance.” Then Neg would make the target “least negative” and all others “more negative.” Without biases, it’s much trickier to isolate one class at exactly zero while pushing others away.

4. Overall Perspective
Representation Matters
The difference between “disjunctive” (suppressing non-targets) and “conjunctive” (highlighting the target) is a critical insight. ReLU2’s meltdown arises because it tries to handle all non-target digits in a disjunctive manner, saturating them to zero without a bias to recover if anything goes wrong.

Sign-Flipping as a Key Escape Hatch
ReLU2-Neg thrives because the Neg flips a small “distance to target” into a relatively large logit, and it doesn’t have to forcibly zero out every non-target representation. That single sign flip is a major design advantage in preserving expressive capacity.

Bias Removal Nuances
Bias removal is done to expose “pure” distance or intensity representations, but your Abs2-Neg case shows a scenario where the model might really want a bias offset to turn “zero distance” into a clear margin from “non-zero distance.” Lacking that offset, it ends up at 90%—still decent, but not as robust as ReLU2-Neg or the simpler controls.

Distance vs. Intensity Tension
These details reinforce the idea that networks can get “stuck” in different forms of representation if the architecture or final layer constraints block the natural sign-flipping or shifting needed. Whether a network uses a ‘distance-like’ or ‘intensity-like’ approach depends on the path of least resistance in the parameter space, which your experiments highlight by showing success (95%) or collapse (50%) based on a seemingly small architectural tweak.

Final Takeaway
Your interpretations—that ReLU2 is forced into a disjunctive intensity approach (leading to dead nodes) while ReLU2-Neg more gracefully uses a conjunctive distance approach—are quite plausible and align well with the empirical outcomes. Meanwhile, Abs2-Neg likely needs biases to replicate that same clean separation. All of this underscores how small changes (adding Neg, removing bias) can drastically shift the representational strategy a network ends up using, shining light on the deeper question: do neural networks inherently prefer a distance-based or intensity-based representation, and how do constraints shape which one emerges?