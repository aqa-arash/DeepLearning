import numpy as np
from . import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.cache = None

    def forward(self, input_tensor):
        self.cache = {}
        batch_size, channels, input_y, input_x = input_tensor.shape
        
        # Unpack shapes
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        # Calculate output dimensions for "valid" padding 
        # Formula: (Input - Kernel) // Stride + 1
        output_y = (input_y - pool_y) // stride_y + 1
        output_x = (input_x - pool_x) // stride_x + 1

        output_tensor = np.zeros((batch_size, channels, output_y, output_x))

        # Perform Max Pooling
        # Iterating spatially to handle "valid" padding and border elements correctly 
        for b in range(batch_size):
            for c in range(channels):
                for y in range(output_y):
                    for x in range(output_x):
                        # Determine the slice window
                        start_y = y * stride_y
                        end_y = start_y + pool_y
                        start_x = x * stride_x
                        end_x = start_x + pool_x

                        # Extract window and find max
                        window = input_tensor[b, c, start_y:end_y, start_x:end_x]
                        max_val = np.max(window)
                        output_tensor[b, c, y, x] = max_val

                        # Store position of max value for backward pass 
                        # argmax returns linear index, we convert to (y, x) within the window
                        max_idx_linear = np.argmax(window)
                        max_pos_local = np.unravel_index(max_idx_linear, (pool_y, pool_x))
                        
                        # Store absolute position or mask info
                        # Key: (b, c, output_y, output_x) -> Value: (absolute_y, absolute_x)
                        self.cache[(b, c, y, x)] = (start_y + max_pos_local[0], start_x + max_pos_local[1])

        self.input_shape = input_tensor.shape
        return output_tensor

    def backward(self, error_tensor):
        output_error = np.zeros(self.input_shape)
        batch_size, channels, output_y, output_x = error_tensor.shape

        # Iterate over the error tensor and route gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for y in range(output_y):
                    for x in range(output_x):
                        # Retrieve the position of the max value from forward pass
                        max_pos = self.cache[(b, c, y, x)]
                        
                        # Accumulate gradient (+= is safer than = if overlaps occur, though rarely for maxpool)
                        output_error[b, c, max_pos[0], max_pos[1]] += error_tensor[b, c, y, x]

        return output_error