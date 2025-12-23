import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.predicted_tensor = []

    def forward(self, predicted_tensor, label_tensor):
        # Use machine epsilon for numerical stability
        epsilon = np.finfo(np.float64).eps
        clipped_predictions = np.clip(predicted_tensor, epsilon, 1. - epsilon)
        
        # Calculate cross-entropy loss
        sample_losses = -np.sum(label_tensor * np.log(clipped_predictions), axis=1)
        
        # Return the SUM (accumulated) loss
        sum_loss = np.sum(sample_losses) 
        
        #store predicted tensor for backward pass
        self.predicted_tensor = predicted_tensor
        return sum_loss

    def backward(self, label_tensor):
        epsilon = np.finfo(np.float64).eps
        clipped_predictions = np.clip(self.predicted_tensor, epsilon, 1. - epsilon)
        # Calculate gradient
        input_gradient = - (label_tensor / clipped_predictions) 
        return input_gradient