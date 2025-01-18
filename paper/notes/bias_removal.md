A single linear layer can represent either distance or intensity metrics just by negating its weights. So just looking at the weights alone won't tell us which representation the network "prefers" to learn. See the following equation:

linear->neg: y = -(Wx+b) = -Wx - b = (-W)x - b

This shows that a linear layer by itself can learn a distance or intensity representation by negating its weights.

Suppose that the network has a bias for learning distance representations. It could learn a positive distance representation (y_d=W_d⋅x + b_d) and then convert it to a negative intensity representation by negating the weights (y_i = -W_d⋅x - b_d).

Then it could make that positive by adding a positive bias to make it a positive intensity metric.

y_i = -W_d*x - b_d + b_i


Removing the bias forces the network to learn either a positive distance or a positive intensity representation.