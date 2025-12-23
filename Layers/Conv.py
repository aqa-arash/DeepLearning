import numpy as np
from scipy import signal
import copy
from . import Base

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.num_kernels = num_kernels
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        
        if len(convolution_shape) == 2:
            self.spatial_dims = 1
            self.input_channels = convolution_shape[0]
            self.kernel_shape = convolution_shape[1:]
        else:
            self.spatial_dims = 2
            self.input_channels = convolution_shape[0]
            self.kernel_shape = convolution_shape[1:]
            
        if isinstance(self.stride_shape, int):
            self.stride_shape = (self.stride_shape,) * self.spatial_dims

        self.weights = np.random.uniform(0, 1, (self.num_kernels, self.input_channels) + self.kernel_shape)
        self.bias = np.random.uniform(0, 1, (self.num_kernels,))
        
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.kernel_shape) * self.input_channels
        fan_out = np.prod(self.kernel_shape) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    def forward(self, input_tensor):
        self.orig_input_shape = input_tensor.shape
        # If input is 2D: (batch, features)
        if input_tensor.ndim == 2:
            # treat features as channels, spatial dims = 1x1
             input_tensor = input_tensor.reshape(input_tensor.shape[0],
                                                 input_tensor.shape[1],
                                                 1, 1)
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        output_shape = (batch_size, self.num_kernels) + input_tensor.shape[2:]
        output = np.zeros(output_shape)

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for c in range(self.input_channels):
                    output[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output[b, k] += self.bias[k]

        if self.spatial_dims == 1:
            return output[:, :, ::self.stride_shape[0]]
        else:
            return output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

    def backward(self, error_tensor):
        # 1. Handle Stride (Upsample error tensor)
        batch_size = error_tensor.shape[0]
        upsampled = np.zeros((batch_size, self.num_kernels) + self.input_tensor.shape[2:])

        if self.spatial_dims == 1:
            upsampled[:, :, ::self.stride_shape[0]] = error_tensor
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
        else:
            upsampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # 2. Calculate Gradient w.r.t Weights
        self.gradient_weights = np.zeros_like(self.weights)
        
        # --- Fixed Padding Logic for Arbitrary Kernel Sizes ---
        if self.spatial_dims == 2:
            ky, kx = self.kernel_shape
            
            # Total padding needed to ensure 'valid' correlation output equals kernel size
            # is (KernelSize - 1)
            pad_y_total = ky - 1
            pad_y_before = pad_y_total // 2
            pad_y_after = pad_y_total - pad_y_before
            
            pad_x_total = kx - 1
            pad_x_before = pad_x_total // 2
            pad_x_after = pad_x_total - pad_x_before
            
            pad_width = ((0,0), (0,0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after))
            
        else: # spatial_dims == 1
            kx = self.kernel_shape[0]
            
            pad_x_total = kx - 1
            pad_x_before = pad_x_total // 2
            pad_x_after = pad_x_total - pad_x_before
            
            pad_width = ((0,0), (0,0), (pad_x_before, pad_x_after))

        # Pad the input
        padded_input = np.pad(self.input_tensor, pad_width, mode='constant')

        for k in range(self.num_kernels):
            for c in range(self.input_channels):
                grad = np.zeros(self.kernel_shape)
                for b in range(batch_size):
                    # Correlate padded input with upsampled error using 'valid'
                    grad += signal.correlate(padded_input[b, c], upsampled[b, k], mode='valid')
                self.gradient_weights[k, c] = grad

        # 3. Calculate Gradient w.r.t Input (Next Error)
        next_error = np.zeros_like(self.input_tensor)
        for b in range(batch_size):
            for c in range(self.input_channels):
                for k in range(self.num_kernels):
                    # Convolve error with weights (equivalent to correlate with flipped weights)
                    next_error[b, c] += signal.convolve(upsampled[b, k], self.weights[k, c], mode='same')

        # 4. Update Weights (Optimization)
        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        # 5. Restore shape if necessary
        if len(self.orig_input_shape) == 2:
            next_error = next_error.reshape(self.orig_input_shape)
            
        # Ensure this return statement is NOT indented inside an if block
        return next_error