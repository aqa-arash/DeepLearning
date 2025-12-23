import numpy as np
from . import Base

class ReLU (Base.BaseLayer):
    def __init__(self):
        super().__init__()
         
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, input_tensor)
        return output_tensor

    def backward(self, error_tensor):
        input_gradient = error_tensor * (self.input_tensor > 0)
        return input_gradient