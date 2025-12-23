import numpy as np 


class Constant :
    def __init__(self, value = 0.1) : 
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Fill weights with a constant value.
        """
        return np.full(weights_shape, self.value, dtype=np.float64)


class UniformRandom: 
    def __init__(self) : 
        pass 

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Uniform random values in [0, 1).
        """
        return np.random.rand(*weights_shape).astype(np.float32)


class Xavier:
    def __init__(self) : 
        pass 

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Glorot/Xavier uniform initialization:
        U(-limit, limit) with limit = sqrt(6 / (fan_in + fan_out))
        """
        denom = fan_in + fan_out
        if denom <= 0:
           sigma = 0.0
        else:
           sigma = np.sqrt(2.0 / denom)
            
        # Use np.random.normal instead of uniform
        # loc=0.0 is the mean, scale=sigma is the standard deviation
        return np.random.normal(loc=0.0, scale=sigma, size=weights_shape).astype(np.float32)

class He: 
    def __init__(self) : 
        pass 

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        He (Kaiming) normal initialization for ReLU:
        N(0, std^2) with std = sqrt(2 / fan_in)
        """
        if fan_in <= 0:
            std = 0.0
        else:
            std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, std, size=weights_shape).astype(np.float32)
