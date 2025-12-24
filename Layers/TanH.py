import numpy as np
from . import Base

class TanH(Base.BaseLayer):
    """
    TanH Activation Layer
    """

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        tanh_derivative = 1 - self.activations ** 2
        return error_tensor * tanh_derivative