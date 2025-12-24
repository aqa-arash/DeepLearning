from Layers import *
from Optimization import *
import copy
import numpy as np

# We assume necessary classes like Optimizer, layers, etc., are defined elsewhere.

class NeuralNetwork:
    """
    A class representing a simple neural network.
    """

    def __init__(self, optimizer, weight_initializer=None, bias_initializer=None):
        """
        Initializes the NeuralNetwork.

        Args:
            optimizer: An optimizer object used for training trainable layers.
        """
        # 1. Implement five member variables
        self.optimizer = optimizer
        self.loss = []       # List to store loss values for each iteration
        self.layers = []     # List to hold the network architecture
        self.data_layer = None   # Provides input data and labels
        self.loss_layer = None   # Special layer providing loss and prediction
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.loss_regularizer = 0.0  # Regularization loss
        
        # Helper variable to store labels between forward and backward pass
        self._current_labels = None
        self._phase = 'train'  # 'train' or 'test'

    def forward(self):
        """
        Performs a forward pass through the network.
        
        Uses input from the data_layer, passes it through all layers,
        and returns the output of the final layer (the loss layer).
        """
        self.loss_regularizer = 0.0
        # Get input and labels from the data layer
        input_tensor, label_tensor = self.data_layer.next()
        
        # Store labels for the backward pass
        self._current_labels = label_tensor
        
        # Propagate the input tensor through all layers
        activation = input_tensor
        for layer in self.layers:
            activation = layer.forward(activation)
            if hasattr(layer,'optimizer') and hasattr(layer.optimizer,'regularizer')\
                  and layer.optimizer.regularizer is not None and  hasattr(layer, 'weights'):
                self.loss_regularizer += layer.optimizer.regularizer.norm(layer.weights)
            
        # Return the output of the last layer (loss)
        loss = self.loss_layer.forward(activation,self._current_labels)
        return loss + self.loss_regularizer

    def backward(self):
        """
        Performs a backward pass through the network.
        
        Starts from the loss layer, passing it the labels, and propagates
        the error gradient back through the network layers in reverse order.
        """
        # Start backpropagation from the loss layer
        error_tensor = self.loss_layer.backward(self._current_labels)
        
        # Propagate the error backward through the layers in reverse,
        # skipping the last layer (loss_layer) which we just computed.
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        """
        Appends a new layer to the network architecture.
        
        If the layer is trainable, it is assigned a deep copy of the
        network's optimizer.
        
        Args:
            layer: The layer object to append.
        """
        # Check if the layer is trainable (assuming it has a 'trainable' attribute)
        if hasattr(layer, 'trainable') and layer.trainable:
            # Make a deep copy of the optimizer for this layer
            optimizer_copy = copy.deepcopy(self.optimizer)
            # Set the optimizer for the layer
            layer.optimizer = optimizer_copy
            layer.initialize(self.weight_initializer, self.bias_initializer)
            
        # Append the layer to the list
        self.layers.append(layer)

    def train(self, iterations):
        """
        Trains the network for a specified number of iterations.
        
        For each iteration, it performs a forward pass, stores the loss,
        and performs a backward pass.
        
        Args:
            iterations (int): The number of iterations to train.
        """
        self.phase= 'train'
        for _ in range(iterations):
            # 1. Perform forward pass which already computes and stores loss in the loss layer
            loss_value = self.forward()
            # 2. Store the loss
            self.loss.append(loss_value)
            # 3. Perform backward pass to compute gradients and update parameters
            self.backward()

    def test(self, input_tensor):
        """
        Propagates an input tensor through the network for testing.
        
        Returns the prediction of the last layer before the loss layer
        (e.g., the output of the SoftMax layer).
        
        Args:
            input_tensor: The input data to test.
            
        Returns:
            The network's prediction (output of the second-to-last layer).
        """
        self.phase='test'
        activation = input_tensor
        
        # Propagate through all layers (the loss layer is not part of self.layers)
        for layer in self.layers:
            activation = layer.forward(activation)
            
        # Return the output of the final layer before the loss
        return activation
    
    @property
    def phase (self):
        """
        Gets the current phase of the network (training or testing).
        
        Returns:
            str: 'train' if in training mode, 'test' if in testing mode.
        """
        return self._phase
    
    @phase.setter
    def phase(self, phase):
        """
        Sets the current phase of the network.
        
        Args:
            phase (str): 'train' to set training mode, 'test' for testing mode.
        """
        if phase not in ['train', 'test']:
            raise ValueError("Phase must be either 'train' or 'test'.")
        self._phase = phase
        for layer in self.layers:
            if hasattr(layer, 'testing_phase'):
                if phase == 'train':
                    layer.testing_phase = False
                else:
                    layer.testing_phase = True
    