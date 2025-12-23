import numpy as np
import Constraints

class Optimizer:
    def __init__ (self) :
        self.regularizer = None 
        pass 
    
    def set_regularizer(self, regularizer) :
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor) :
        raise NotImplementedError("This method should be overridden by subclasses.")


class Sgd(Optimizer):
    def __init__ (self, learningrate): 
        super().__init__()
        self.learningrate = learningrate
    
    def calculate_update(self, weight_tensor, gradient_tensor) :
        if self.regularizer is not None :
            gradient_tensor = gradient_tensor +  self.regularizer.calculate_gradient(weight_tensor)
        updated_weight = weight_tensor - self.learningrate * gradient_tensor
        return updated_weight
    

class SgdWithMomentum(Optimizer):
    def __init__ (self, learning_rate, momentum_rate = 0.9) : 
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self._velocity = None 
    
    def calculate_update(self, weight_tensor, gradient_tensor) :
        if self._velocity is None : 
            self._velocity = np.zeros_like(weight_tensor)

        if self.regularizer is not None :
            gradient_tensor = gradient_tensor +  self.regularizer.calculate_gradient(weight_tensor)
        
        # Update velocity
        self._velocity = self.momentum_rate * self._velocity - self.learning_rate * gradient_tensor
        
        # Update weights
        weight_tensor += self._velocity
        
        return weight_tensor
    
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
        
        if self.regularizer is not None :
            gradient_tensor = gradient_tensor +  self.regularizer.calculate_gradient(weight_tensor)
        
        if self._m is None : 
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
        
        # Update weights
        weight_tensor -= self.learningrate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return weight_tensor