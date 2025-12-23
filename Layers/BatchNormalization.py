import numpy as np
import Base

class BatchNormalization(Base.BaseLayer):
    """
    Batch Normalization Layer
    """

    def __init__(self, channels, optimizer = None,  epsilon=1e-10, momentum=0.9):
        super().__init__()
        self.trainable = True
        self.epsilon = epsilon
        self.momentum = momentum
        self.channels = channels
        self.original_shape = None
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.cache = None
        self._optimizer = optimizer

    def initialize(self):
        #always initializes weights with 1 and biases with 0
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor):
        if self.gamma is None:
            self.gamma = np.ones(input_tensor.shape[1])
        if self.beta is None:
            self.beta = np.zeros(input_tensor.shape[1])
        if self.running_mean is None:
            self.running_mean = np.zeros(input_tensor.shape[1])
        if self.running_var is None:
            self.running_var = np.zeros(input_tensor.shape[1])

        input_tensor = self.reformat(input_tensor)

        if self._phase == 'train':
            batch_mean = np.mean(input_tensor, axis=0)
            batch_var = np.var(input_tensor, axis=0)

            normalized_input = (input_tensor - batch_mean) / np.sqrt(batch_var + self.epsilon)
            output = self.gamma * normalized_input + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cache = (input_tensor, normalized_input, batch_mean, batch_var)
        else:
            normalized_input = (input_tensor - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            output = self.gamma * normalized_input + self.beta
       
        output = self.reformat(output)
        return output
    
    def backward(self, error_tensor):
        input_tensor, normalized_input, batch_mean, batch_var = self.cache
        m = input_tensor.shape[0]

        error_tensor = self.reformat(error_tensor)

        dbeta = np.sum(error_tensor, axis=0)
        dgamma = np.sum(error_tensor * normalized_input, axis=0)

        dnormalized = error_tensor * self.gamma
        dvar = np.sum(dnormalized * (input_tensor - batch_mean) * -0.5 * (batch_var + self.epsilon) ** -1.5, axis=0)
        dmean = np.sum(dnormalized * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (input_tensor - batch_mean), axis=0)

        dinput = dnormalized / np.sqrt(batch_var + self.epsilon) + dvar * 2 * (input_tensor - batch_mean) / m + dmean / m

        if self._optimizer is not None:
            self.gamma -= self._optimizer.calculate_update(self.gamma, dgamma)
            self.beta -= self._optimizer.calculate_update(self.beta, dbeta)

        dinput = self.reformat(dinput)
        return dinput
    
    def reformat (self, tensor) :
        if len(tensor.shape) > 2 :
            # Flatten all dimensions except the first (batch size)
            self.original_shape = tensor.shape
            return tensor.reshape(tensor.shape[0], -1)
        else :
            if self.original_shape is not None :
                # Restore the original shape
                reshaped_tensor = tensor.reshape(self.original_shape)
                self.original_shape = None
                return reshaped_tensor
            else : # increase the dimentsionality to 4D
                return  tensor.reshape(tensor.shape[0], tensor.shape[1], 1, 1)
            
    @property
    def optimizer(self):
        return self._optimizer  
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
