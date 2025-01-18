import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from train_models import evaluate_all_models

# helps prevent vanishing gradient with large negative exponents in LogSoftmax
class MedianOffset(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, num_features]
        
        Returns:
            out shape: [batch_size, num_features]
        """
        # Compute median along the features dimension, for each sample in the batch.
        # median() returns (values, indices), so we take .values here.
        med = x.median(dim=1, keepdim=True).values
        
        # Subtract the median from each feature so that the per-sample median becomes zero
        out = x - med
        return out

class L2Norm(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Each output node gets its own set of reference points
        self.x0 = nn.Parameter(torch.empty(out_features, in_features))
        self.alpha = nn.Parameter(torch.ones(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize x0 to zero
        # nn.init.zeros_(self.x0)
        nn.init.normal_(self.x0, std=0.1)  # or use xavier/kaiming init
        with torch.no_grad():
            self.alpha.fill_(1.0)  # start with alpha=1 for all
    
    def forward(self, x):
        """
        x shape: [batch_size, in_features]
        self.x0 shape: [out_features, in_features]
        self.alpha shape: [out_features, in_features]
        
        We want an output shape: [batch_size, out_features]
        where out[b, j] = sqrt( sum_i alpha[j, i] * (x[b,i] - x0[j,i])^2 ).
        """
        # Suppose alpha, x0 each has shape [out_features, in_features].
        # x has shape [batch_size, in_features].

        # For each class j:
        #   dist_sq[b, j] = (x[b]^2 * alpha[j]).sum() 
        #                 + (x0[j]^2 * alpha[j]).sum()
        #                 - 2 * (x[b] * x0[j] * alpha[j]).sum()

        # We can do it in batched form if we’re careful.

        x_sq = x**2          # [batch_size, in_features]
        x0_sq = self.x0**2   # [out_features, in_features]

        # Weighted X squares:
        #   alpha has shape [out_features, in_features]
        #   we want a single matmul or so that yields [batch_size, out_features]

        # 1) Weighted x^2 -> shape [batch_size, in_features, 1]?
        #    Not straightforward to do a single matmul with alpha unless we reshape carefully.

        # A more direct approach:
        # dist_sq = (x_sq @ alpha.T) + (x0_sq * alpha).sum(dim=1) - 2*((x * x0) * alpha).sum(dim=1)
        # But we need the (x*x0) part in a batched sense. 
        # That can be done via a trick: (x * x0[j]) is basically x @ diag(x0[j]) if x0[j] is diagonal,
        # but we don't want a full diag approach either.

        # Let's do them in pieces:

        # Weighted sum of x^2:
        weighted_x_sq = x_sq @ self.alpha.T  # shape [batch_size, out_features]

        # Weighted sum of x0^2:
        weighted_x0_sq = (x0_sq * self.alpha).sum(dim=1).unsqueeze(0)  # shape [1, out_features]

        # Weighted cross term:
        # cross_term[b, j] = (x[b] * x0[j]) * alpha[j], summed over in_features
        #   => cross_term[b, j] = ( x * ( x0[j] * alpha[j] ) ).sum(dim=1)
        # We can do a matmul if we define a "weighted_x0" = x0 * alpha.
        weighted_x0 = self.x0 * self.alpha  # shape [out_features, in_features]
        # cross_term[b, j] = x[b] @ weighted_x0[j].T
        cross_term = x @ weighted_x0.T  # shape [batch_size, out_features]

        dist_sq = weighted_x_sq + weighted_x0_sq - 2.0 * cross_term
        # [batch_size, out_features]

        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
        return dist

class InputBiasLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Each output node gets its own set of reference points
        self.x0 = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize x0 to zero
        nn.init.zeros_(self.x0)
    
    def forward(self, x):
        # x shape: [batch_size, in_features]
        # x0 shape: [out_features, in_features]
        # For each output node, subtract its x0 values from input and sum
        # Unsqueeze x to [batch_size, 1, in_features] for broadcasting
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, in_features]
        # Subtract x0 and sum along last dimension
        # print(f"x_expanded.shape = {x_expanded.shape}")
        # print(f"self.x0.shape = {self.x0.shape}")
        y = (x_expanded - self.x0).sum(dim=2)  # Returns [batch_size, out_features]
        # print(f"y = {y.shape}")
        return y


class PointSlopeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.x0 = nn.Parameter(torch.empty(out_features, in_features))  # Each output has its own x0
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using same strategy as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        # Initialize x0 to zero
        nn.init.zeros_(self.x0)
    
    def forward(self, x):
        # x shape: [batch_size, in_features]
        # Expand x for broadcasting: [batch_size, 1, in_features]
        x_expanded = x.unsqueeze(1)
        # Subtract each node's x0 from input: [batch_size, out_features, in_features]
        centered = x_expanded - self.x0
        # Multiply by weights and sum along last dimension
        return (centered * self.weight).sum(dim=2)

class Model_ReLU_Bias(nn.Module):
    def __init__(self):
        super(Model_ReLU_Bias, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU_Bias"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear(with bias) -> y"

class Model_ReLU2_Bias(torch.nn.Module):
    def __init__(self):
        super(Model_ReLU2_Bias, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(128, 10)  # Added bias back
        self.activation1 = torch.nn.ReLU()
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.activation1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU2-Bias"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear(with bias) -> ReLU -> y"

class Model_ReLU2_Neg_Bias(nn.Module):
    def __init__(self):
        super(Model_ReLU2_Neg_Bias, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)
        self.activation1 = nn.ReLU()
        self.neg = models.Neg()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.activation1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU2-Neg"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear(with bias) -> ReLU -> Neg -> y"

class Model_Abs2_Neg_Bias(torch.nn.Module):
    def __init__(self):
        super(Model_Abs2_Neg_Bias, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = models.Abs()
        self.linear1 = torch.nn.Linear(128, 10, bias=True)
        self.activation1 = models.Abs()
        self.neg = models.Neg()
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.activation1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs2-Neg-Bias"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear(with bias) -> Abs -> Neg -> y"

class Model_ReLU2_PointSlope(torch.nn.Module):
    def __init__(self):
        super(Model_ReLU2_PointSlope, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = torch.nn.ReLU()
        self.linear1 = PointSlopeLinear(128, 10)
        self.activation1 = torch.nn.ReLU()
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.activation1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU2-PointSlope"
    
    def description(self):
        return "x -> Linear -> ReLU -> PointSlope -> ReLU -> y"

class Model_Abs_Bias(nn.Module):
    def __init__(self):
        super(Model_Abs_Bias, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = models.Abs()
        self.linear1 = nn.Linear(128, 10, bias=True)
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs-Bias"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear(with bias) -> y"

class Model_Abs2_Bias(nn.Module):
    def __init__(self):
        super(Model_Abs2_Bias, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = models.Abs()
        self.linear1 = nn.Linear(128, 10, bias=True)
        self.activation1 = models.Abs()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.activation1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs2-Bias"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear(with bias) -> Abs -> y"

class Model_Abs_L2(torch.nn.Module):
    def __init__(self):
        super(Model_Abs_L2, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = models.Abs()
        self.linear1 = L2Norm(128, 10)
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs-L2"
    
    def description(self):
        return "x -> Linear -> Abs -> L2Norm -> y"

class Model_Abs_L2_Neg(torch.nn.Module):
    def __init__(self):
        super(Model_Abs_L2_Neg, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = models.Abs()
        self.linear1 = L2Norm(128, 10)
        self.neg = models.Neg()
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs-L2-Neg"
    
    def description(self):
        return "x -> Linear -> Abs -> L2Norm -> Neg -> y"

class Model_ReLU_L2(torch.nn.Module):
    def __init__(self):
        super(Model_ReLU_L2, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = L2Norm(128, 10)
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU-L2"
    
    def description(self):
        return "x -> Linear -> ReLU -> L2Norm -> y"

class Model_ReLU_L2_Neg(torch.nn.Module):
    def __init__(self):
        super(Model_ReLU_L2_Neg, self).__init__()
        self.linear0 = torch.nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = L2Norm(128, 10)
        self.neg = models.Neg()
        self.layers = torch.nn.Sequential(
            self.linear0,
            self.activation0,
            self.linear1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU-L2-Neg"
    
    def description(self):
        return "x -> Linear -> ReLU -> L2Norm -> Neg -> y"

def create_model_list(device):
    return [
        # models.Model_ReLU2().to(device),  
        # Model_ReLU2_Bias().to(device),
        models.Model_Abs2().to(device),  
        models.Model_Abs2_Neg().to(device),  
        # Model_Abs2_Neg_Bias().to(device),
        # Model_Abs2_Neg_L2Norm().to(device),
        # Model_ReLU2_PointSlope().to(device),
    ]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("No GPU found. Exiting...")
        sys.exit(1)

    # Create and evaluate models
    model_list = create_model_list(device)
    # evaluate_all_models({models.Model_Abs2().to(device), models.Model_Abs2_Neg().to(device),}, runs_per_model=1, epochs=5000, lr=0.001, exp_name="models_with_bias")
    # evaluate_all_models({Model_Abs2_Neg_L2().to(device),}, runs_per_model=20, epochs=5000, lr=0.001, exp_name="models_with_bias")
    # evaluate_all_models({Model_ReLU_Neg_L2().to(device),}, runs_per_model=20, epochs=5000, lr=0.001, exp_name="models_with_bias")
    # evaluate_all_models({
    #     Model_ReLU_Bias().to(device), 
    #     Model_ReLU2_Bias().to(device), 
    #     Model_ReLU2_Neg_Bias().to(device), 
    #     Model_ReLU_L2_Neg().to(device),
    #     Model_Abs_Bias().to(device),
    #     Model_Abs2_Bias().to(device),
    #     Model_Abs2_Neg_Bias().to(device),
    #     Model_Abs_L2_Neg().to(device),
    #     }, runs_per_model=20, epochs=50000, lr=0.001, exp_name="models_with_bias")
    evaluate_all_models({
        Model_ReLU_L2().to(device),
        Model_Abs_L2().to(device),
        }, runs_per_model=20, epochs=50000, lr=0.001, exp_name="models_with_bias")