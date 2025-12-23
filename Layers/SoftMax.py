import numpy as np
from . import Base

class SoftMax (Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # Subtract max for numerical stability
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_tensor = probabilities
        #store input for backward pass
        self.input_tensor = input_tensor
        return self.output_tensor

    def backward(self, error_tensor):
        # Initialize gradient tensor
        num_samples, num_classes = self.output_tensor.shape
        input_gradient = np.zeros((num_samples, num_classes))

        # Compute the Jacobian matrix for each sample and multiply by the error tensor
        # Vectorized Jacobian-vector product: J * e = y * (e - (y·e))
        s = np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True)  # (num_samples, 1)
        input_gradient = self.output_tensor * (error_tensor - s)  # broadcasted elementwise

        return input_gradient