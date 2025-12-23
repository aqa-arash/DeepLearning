import numpy as np
from . import Base
np.random.seed(42)


class FullyConnected (Base.BaseLayer):
    def __init__(self, input_size, output_size, optimizer=None):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = optimizer
        
        # Initialize weights and biases
        self.weights = np.random.rand(input_size +1, output_size)
        self.weights*=0.2 
         
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        # Create the extended input tensor by appending a column of ones
        extended_input = np.hstack((input_tensor, np.ones((batch_size, 1))))
        # Store this for the backward pass
        self.input_tensor = extended_input
        # Perform the dot product
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer


    def backward(self, error_tensor):
        # Calculate gradients
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        
        # Return the gradient to propagate to the previous layer
        input_gradient = np.dot(error_tensor, self.weights.T)

        # Update weights and biases using the optimizer
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            
        # Return the gradient *without* the bias column
        return input_gradient[:, :-1]
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size
        fan_out = self.output_size
        
        # Initialize weights (excluding bias row)
        self.weights[:-1, :] = weights_initializer.initialize(
            (self.input_size, self.output_size), fan_in, fan_out)
        
        # Initialize biases (last row)
        self.weights[-1, :] = bias_initializer.initialize(
            (1, self.output_size), fan_in, fan_out)

