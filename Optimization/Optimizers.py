import numpy as np
from . import Constraints

class Optimizer:
    def __init__ (self) :
        self.regularizer = None 
        pass 
    
    def add_regularizer(self, regularizer) :
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor) :
        raise NotImplementedError("This method should be overridden by subclasses.")


class Sgd(Optimizer):
    def __init__ (self, learningrate): 
        super().__init__()
        self.learningrate = learningrate
    
    def calculate_update(self, weight_tensor, gradient_tensor) :
        shrinked_weights = weight_tensor.copy()
        if self.regularizer is not None :
            shrinked_weights = weight_tensor - self.learningrate *  self.regularizer.calculate_gradient(weight_tensor)
        updated_weight = shrinked_weights - self.learningrate * gradient_tensor
        return updated_weight
    

class SgdWithMomentum(Optimizer):
    def __init__ (self, learning_rate, momentum_rate = 0.9) : 
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self._velocity = None 
    
    def calculate_update(self, weight_tensor, gradient_tensor) :
        
        shrinked_weights = weight_tensor.copy()
        if self._velocity is None : 
            self._velocity = np.zeros_like(weight_tensor)

        if self.regularizer is not None :
            shrinked_weights = weight_tensor - self.learning_rate *  self.regularizer.calculate_gradient(weight_tensor)
        
        # Update velocity
        self._velocity = self.momentum_rate * self._velocity - self.learning_rate * gradient_tensor
        
        # Update weights
        shrinked_weights += self._velocity
        
        return shrinked_weights
    
class Adam(Optimizer):
    def __init__ (self, learningrate, mu = 0.9, rho = 0.999) : 
        super().__init__()

        self.learningrate = learningrate
        self.mu = mu
        self.rho = rho
        self.epsilon =  1e-8
        self._m = None 
        self._v = None 
        self._t = 0 
    
    def calculate_update(self, weight_tensor, gradient_tensor) :
        # Use shrinked weights if regularizer is present (do NOT add reg gradient to gradient_tensor)
        if self.regularizer is not None:
            shrinked_weights = weight_tensor - self.learningrate* self.regularizer.calculate_gradient(weight_tensor)
        else:
            shrinked_weights = weight_tensor

        if self._m is None:
            self._m = np.zeros_like(weight_tensor)
            self._v = np.zeros_like(weight_tensor)

        self._t += 1
        # Update biased first moment estimate
        self._m = self.mu * self._m + (1 - self.mu) * gradient_tensor

        # Update biased second raw moment estimate
        self._v = self.rho * self._v + (1 - self.rho) * (gradient_tensor ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self._m / (1 - self.mu ** self._t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self._v / (1 - self.rho ** self._t)

        # Compute updated weights (do not update in-place)
        updated_weight = shrinked_weights - self.learningrate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_weight