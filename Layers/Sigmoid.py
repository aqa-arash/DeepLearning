import numpy as np
from . import Base

class Sigmoid(Base.BaseLayer):
    """
    Sigmoid Activation Layer
    """

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.activations = None

    def forward(self, input_tensor):
        self.activations =  1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        sigmoid_derivative = self.activations * (1 - self.activations)
        return error_tensor * sigmoid_derivative