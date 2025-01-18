import torch
import torch.nn as nn

def create_model_list(device):
    return [
        Model_ReLU().to(device),
        Model_Abs().to(device),
        Model_ReLU2().to(device),
        Model_Abs2().to(device),
        Model_ReLU2_Neg().to(device),
        Model_Abs2_Neg().to(device),
    ]


class PerturbationLayer(nn.Module):
    """
    A layer that can modify the minimum and maximum values of activations.
    Can perturb either a single node or the entire layer.
    During training, this layer acts as an identity function.   
    """
    def __init__(self):
        super().__init__()
        self.reset_perturbation()
    
    def forward(self, x):
        """Apply min/max perturbation to input tensor."""
        # Check if perturbation is active
        if self.min_val == float('-inf') and self.max_val == float('inf'):
            return x
            
        if self.node_idx is None:
            # Perturb all nodes
            return torch.clamp(x, min=self.min_val, max=self.max_val)
        else:
            # Perturb single node
            output = x
            output[:, self.node_idx] = torch.clamp(
                output[:, self.node_idx], 
                min=self.min_val, 
                max=self.max_val
            )
            return output
    
    def set_perturbation(self, min_val=float('-inf'), max_val=float('inf'), node_idx=None):
        """
        Set perturbation parameters.
        
        Args:
            min_val (float, optional): New minimum value. Default: -inf
            max_val (float, optional): New maximum value. Default: inf
            node_idx (int, optional): Index of node to perturb. If None, perturbs all nodes.
        """
        self.min_val = min_val
        self.max_val = max_val
        
        self.node_idx = node_idx
    
    def reset_perturbation(self):
        """Reset to identity function (no perturbation)."""
        self.min_val = float('-inf')
        self.max_val = float('inf')
        self.node_idx = None
    
    def get_perturbation(self):
        """Get current perturbation parameters."""
        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'node_idx': self.node_idx
        }
        

class Neg(nn.Module):
    def forward(self, x):
        return -x


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


# Define the MLP model
class Model_ReLU(nn.Module):
    def __init__(self):
        super(Model_ReLU, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear -> y"


class Model_Abs(nn.Module):
    def __init__(self):
        super(Model_Abs, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = Abs()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear -> y"


class Model_ReLU2(nn.Module):
    def __init__(self):
        super(Model_ReLU2, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.activation1 = nn.ReLU()
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1,
            self.activation1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU2"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear -> ReLU -> y"


class Model_Abs2(nn.Module):
    def __init__(self):
        super(Model_Abs2, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = Abs()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.activation1 = Abs()
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1,
            self.activation1
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs2"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear -> Abs -> y"


class Model_ReLU2_Neg(nn.Module):
    def __init__(self):
        super(Model_ReLU2_Neg, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.activation1 = nn.ReLU()
        self.neg = Neg()
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1,
            self.activation1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "ReLU2-Neg"
    
    def description(self):
        return "x -> Linear -> ReLU -> Linear -> ReLU -> Neg -> y"


class Model_Abs2_Neg(nn.Module):
    def __init__(self):
        super(Model_Abs2_Neg, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.activation0 = Abs()
        self.linear1 = nn.Linear(128, 10, bias=False)
        self.activation1 = Abs()
        self.neg = Neg()
        self.perturbation = PerturbationLayer()
        self.layers = nn.Sequential(
            self.linear0,
            self.activation0,
            self.perturbation,
            self.linear1,
            self.activation1,
            self.neg
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "Abs2-Neg"
    
    def description(self):
        return "x -> Linear -> Abs -> Linear -> Abs -> Neg -> y"
