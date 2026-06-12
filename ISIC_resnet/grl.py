# src/model/grl.py (Create this new file and directory if needed)
import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer Function.
    Adapted from: https://github.com/fungtion/DANN/blob/master/models/functions.py
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass: Identity function.
        Args:
            ctx: Context object to save information for backward pass.
            x: Input tensor.
            lambda_: Scaling factor for the reversed gradient.
        """
        ctx.lambda_ = lambda_  # Store lambda for backward pass
        return x.view_as(x) # Return input tensor unchanged

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Multiply gradient by -lambda_.
        Args:
            ctx: Context object with saved lambda_.
            grad_output: Gradient from the subsequent layer.
        """
        # Retrieve lambda and calculate reversed gradient
        lambda_ = ctx.lambda_
        reversed_grad = grad_output.neg() * lambda_
        # The gradient is passed only wrt the input x.
        # The lambda_ parameter itself doesn't require gradients.
        return reversed_grad, None # Gradient wrt x, None for lambda_

class GradientReversalLayer(torch.nn.Module):
    """
    Gradient Reversal Layer Module.
    Applies the GradientReversalFunction.

    Args:
        lambda_ (float): The scaling factor for the gradient reversal. Default is 1.0.
    """
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """Applies the gradient reversal function."""
        return GradientReversalFunction.apply(x, self.lambda_)