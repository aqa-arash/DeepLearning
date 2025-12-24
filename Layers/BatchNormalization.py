import numpy as np
from . import Base

class BatchNormalization(Base.BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # Hyperparameters
        self.epsilon = 1e-10
        self.momentum = 0.8
        self._optimizer = None
        
        # Learnable Parameters
        self.gamma = None
        self.beta = None
        
        # Fixed Statistics (Running Averages)
        self.running_mean = None
        self.running_var = None
        
        # Gradients
        self._gradient_gamma = None
        self._gradient_beta = None
        
        # State
        self.cache = None
        self.original_shape = None
        self.is_initialized = False  # Track if running stats have been set
        
        self.initialize()

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)
        # Note: running_mean/var are initialized to 0/1 here to have a valid state,
        # but will be overwritten by the first batch's stats during forward().
        self.running_mean = np.zeros(self.channels)
        self.running_var = np.ones(self.channels)
        self.is_initialized = False

    def reformat(self, tensor):
        """
        Robustly toggles tensor shape between Image (4D) and Vector (2D).
        - If input is 4D (B, C, H, W): Flattens spatial dims -> (B*H*W, C)
        - If input is 2D (N, C): Reshapes back to 4D -> (B, C, H, W) using stored original_shape
        """
        # Case 1: Image -> Vector (Flatten)
        if len(tensor.shape) == 4:
            B, C, H, W = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, C)

        # Case 2: Vector -> Image (Unflatten)
        elif len(tensor.shape) == 2 and self.original_shape is not None and len(self.original_shape) == 4:
            B, C, H, W = self.original_shape
            if tensor.shape[0] == B * H * W:
                return tensor.reshape(B, H, W, C).transpose(0, 3, 1, 2)
            else:
                return tensor
        else:
            return tensor

    def forward(self, input_tensor):
        # 1. Store Shape Information only if input is 4D
        if len(input_tensor.shape) == 4:
            self.original_shape = input_tensor.shape
        else:
            self.original_shape = None

        # 2. Reformat Input (4D -> 2D if necessary)
        input_scrubbed = self.reformat(input_tensor)

        # 3. Calculate Mean and Variance
        if not self.testing_phase:
            batch_mean = np.mean(input_scrubbed, axis=0)
            batch_var = np.var(input_scrubbed, axis=0)

            # Requirement: Initialize running stats with the FIRST batch's stats
            if not self.is_initialized:
                self.running_mean = batch_mean
                self.running_var = batch_var
                self.is_initialized = True
            else:
                # Update Moving Averages
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            mean = batch_mean
            var = batch_var
            self.cache = (input_scrubbed, mean, var)
        else:
            # Test Phase: Use running averages
            mean = self.running_mean
            var = self.running_var

        # 4. Normalize
        normalized = (input_scrubbed - mean) / np.sqrt(var + self.epsilon)

        # 5. Scale and Shift
        output = self.gamma * normalized + self.beta

        # 6. Reformat Output (2D -> 4D if necessary)
        return self.reformat(output)

    def backward(self, error_tensor):
        # 1. Reformat Error (4D -> 2D if necessary)
        error_scrubbed = self.reformat(error_tensor)
        
        input_scrubbed, mean, var = self.cache
        
        # Gradients w.r.t parameters
        # error: (N, C), normalized: (N, C) -> Sum over N -> (C,)
        normalized = (input_scrubbed - mean) / np.sqrt(var + self.epsilon)
        self._gradient_gamma = np.sum(error_scrubbed * normalized, axis=0)
        self._gradient_beta = np.sum(error_scrubbed, axis=0)

        # Gradient w.r.t input
        # Standard Batch Norm Backward Pass
        N = input_scrubbed.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        
        # Helper terms
        d_normalized = error_scrubbed * self.gamma
        d_var = np.sum(d_normalized * (input_scrubbed - mean) * -0.5 * std_inv**3, axis=0)
        d_mean = np.sum(d_normalized * -std_inv, axis=0) + d_var * np.mean(-2.0 * (input_scrubbed - mean), axis=0)
        
        d_input = (d_normalized * std_inv) + (d_var * 2.0 * (input_scrubbed - mean) / N) + (d_mean / N)

        # Update weights if optimizer exists
        if self._optimizer:
            self.gamma = self._optimizer.calculate_update(self.gamma, self._gradient_gamma)
            self.beta = self._optimizer.calculate_update(self.beta, self._gradient_beta)

        # 2. Reformat Gradient (2D -> 4D if necessary)
        return self.reformat(d_input)

    # --- Properties ---
    @property
    def weights(self):
        return self.gamma
    
    @weights.setter
    def weights(self, value):
        self.gamma = value

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, value):
        self.beta = value

    @property
    def gradient_weights(self):
        return self._gradient_gamma

    @property
    def gradient_bias(self):
        return self._gradient_beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def testing_phase(self):
       return getattr(self, '_testing_phase', False)

    @testing_phase.setter
    def testing_phase(self, value):
        self._testing_phase = value