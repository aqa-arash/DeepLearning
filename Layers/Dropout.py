import numpy as np
import Base

class Dropout(Base.BaseLayer):
    """
    Dropout layer for regularization in neural networks.
    During training, randomly sets a fraction of input units to zero
    at each update to prevent overfitting.

    Attributes:
        probability (float): The fraction of the input units to keep.
        mask (np.ndarray): The mask used to drop units during training.
    """

    def __init__(self, probability):
        """
        Initializes the Dropout layer.

        Args:
            probability (float): The fraction of the input units to keep.
        """
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        """
        Forward pass for the Dropout layer.

        Args:
            input_tensor (np.ndarray): The input data.  
        Returns:
            np.ndarray: The output after applying dropout.
            """
        if self._phase == 'train':
            # Create a mask with the same shape as input_tensor
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(input_tensor.dtype)
            # Scale the output to maintain the expected value
            output_tensor = input_tensor * self.mask / self.probability
        else:
            # During testing, no dropout is applied
            output_tensor = input_tensor
        return output_tensor
    

    def backward(self, error_tensor):
        """
        Backward pass for the Dropout layer.
        Args:
            error_tensor (np.ndarray): The gradient of the loss with respect to the output of this layer.
        Returns:
            np.ndarray: The gradient of the loss with respect to the input of this layer.
        """
        if self._phase == 'train':
            # Apply the mask to the output gradient and scale it    
            input_gradient = error_tensor * self.mask /self.probability
        else:   
            # During testing, gradients pass through unchanged
            input_gradient = error_tensor        

        return input_gradient