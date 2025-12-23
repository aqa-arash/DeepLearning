import numpy as np
from . import Base

class Flatten(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        flattened_tensor = input_tensor.reshape(batch_size, -1)
        return flattened_tensor

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)