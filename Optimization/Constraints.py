import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def compute_loss(self, weights):
        return self.alpha * np.sum(np.abs(weights))
    
    def norm(self, weights):
        # L1 norm of weights
        return np.linalg.norm(weights, ord=1)

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)
    


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.enhanced_loss_norm=0

    def compute_loss(self, weights):
        return 0.5 * self.alpha * np.sum(weights ** 2)
    
    def norm(self, weights):
        # L2 norm of weights
        return np.linalg.norm(weights)

    def calculate_gradient(self, weights):
        return self.alpha * weights