import numpy as np
from . import Base
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size, optimizer=None):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self._memorize = False

        # --- Sub-Layers ---
        # 1. Hidden Layer: Takes [Input, Hidden_Prev] -> New Hidden
        # Input dimension is input_size + hidden_size
        self.fc_hidden = FullyConnected(input_size + hidden_size, hidden_size)
        self.tanh = TanH()

        # 2. Output Layer: Takes [Hidden] -> Output
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()

        self.last_hidden_state = None
        self.gradient_weights_n = None # internal storage for the property

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        # Handle "Batch as Time" (single sequence)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor[np.newaxis, :, :] # (1, Time, Features)

        batch_size, seq_length, _ = input_tensor.shape
        self.batch_size = batch_size # Store for backward

        # Initialize hidden state
        if self._memorize and self.last_hidden_state is not None and self.last_hidden_state.shape[0] == batch_size:
            h_t = self.last_hidden_state
        else:
            h_t = np.zeros((batch_size, self.hidden_size))

        output_tensor = np.zeros((batch_size, seq_length, self.output_size))
        
        # Cache for BPTT
        self.cache = {
            'fc_hidden_inputs': [], # Stores input_tensor (with bias) for each step
            'fc_output_inputs': [], # Stores input_tensor (with bias) for each step
            'h_states': [],         # Stores hidden state before activation
            'y_outputs': [],        # Stores output y (for sigmoid derivative)
            'h_activations': [h_t]  # Stores h_t (after activation)
        }

        for t in range(seq_length):
            x_t = input_tensor[:, t, :]
            
            # --- 1. Hidden Step ---
            # Concatenate x_t and h_{t-1}
            x_con = np.concatenate([x_t, h_t], axis=1)
            
            # Pass through FC Hidden
            # Note: FC layer adds the bias column internally. We must capture that state.
            self.fc_hidden.forward(x_con)
            # Store the internal extended input (with bias) used by FC layer
            self.cache['fc_hidden_inputs'].append(self.fc_hidden.input_tensor)
            
            # Pass through TanH
            # We get the pre-activation from FC logic, but FC.forward returns result of dot product
            # So we just pass the result to TanH
            h_out = np.dot(self.fc_hidden.input_tensor, self.fc_hidden.weights) # Re-calculate or assume FC returns it
            # Actually, just use the return value of forward:
            h_out = self.fc_hidden.forward(x_con) # This runs the dot product again, which is fine, or just rely on prev line
            
            h_t = self.tanh.forward(h_out)
            self.cache['h_states'].append(h_t) # Store for debugging/logic
            self.cache['h_activations'].append(h_t)

            # --- 2. Output Step ---
            # Pass through FC Output
            y_out_pre = self.fc_output.forward(h_t)
            # Store the internal extended input (with bias) used by FC layer
            self.cache['fc_output_inputs'].append(self.fc_output.input_tensor)

            # Pass through Sigmoid
            y_t = self.sigmoid.forward(y_out_pre)
            self.cache['y_outputs'].append(y_t)
            
            output_tensor[:, t, :] = y_t

        self.last_hidden_state = h_t.copy()
        
        # Return flattened if input was originally 2D
        if input_tensor.shape[0] == 1:
            return output_tensor[0, :, :]
        return output_tensor

    def backward(self, error_tensor):
        if error_tensor.ndim == 2:
             error_tensor = error_tensor[np.newaxis, :, :]

        batch_size, seq_length, _ = error_tensor.shape
        
        # Accumulators for weights (initialized to zero)
        grad_weights_hidden_acc = np.zeros_like(self.fc_hidden.weights)
        grad_weights_output_acc = np.zeros_like(self.fc_output.weights)
        
        dh_next = np.zeros((batch_size, self.hidden_size))
        grad_input = np.zeros((batch_size, seq_length, self.input_size))

        # Important: Disable optimizer in sub-layers for the loop
        # We perform the update MANUALLY after accumulation
        fc_hidden_opt = self.fc_hidden.optimizer
        fc_output_opt = self.fc_output.optimizer
        self.fc_hidden.optimizer = None
        self.fc_output.optimizer = None

        for t in reversed(range(seq_length)):
            dy = error_tensor[:, t, :]
            
            # --- Output Layer Gradients ---
            # 1. Sigmoid Derivative
            # Restore sigmoid state (activations) to calculate derivative
            self.sigmoid.activations = self.cache['y_outputs'][t]
            dy_sigmoid = self.sigmoid.backward(dy)
            
            # 2. FC Output Layer
            # Restore input tensor state
            self.fc_output.input_tensor = self.cache['fc_output_inputs'][t]
            # Backward propagates error to h_t
            dh_from_output = self.fc_output.backward(dy_sigmoid)
            
            # Accumulate Output Weights Gradient
            grad_weights_output_acc += self.fc_output.gradient_weights

            # --- Hidden Layer Gradients ---
            # Combine gradients flowing to h_t: from output + from next time step
            dh_total = dh_from_output + dh_next
            
            # 1. TanH Derivative
            # Restore tanh state (activations)
            self.tanh.activations = self.cache['h_activations'][t+1] # index t+1 corresponds to h_t
            dtanh = self.tanh.backward(dh_total)
            
            # 2. FC Hidden Layer
            # Restore input tensor state
            self.fc_hidden.input_tensor = self.cache['fc_hidden_inputs'][t]
            # Backward propagates error to [x_t, h_{t-1}]
            d_concat = self.fc_hidden.backward(dtanh)
            
            # Accumulate Hidden Weights Gradient
            grad_weights_hidden_acc += self.fc_hidden.gradient_weights
            
            # Split Gradient: [Input (x_t) | Previous Hidden (h_{t-1})]
            grad_input[:, t, :] = d_concat[:, :self.input_size]
            dh_next = d_concat[:, self.input_size:]

        # Restore optimizers
        self.fc_hidden.optimizer = fc_hidden_opt
        self.fc_output.optimizer = fc_output_opt
        
        # Save accumulated gradients to properties
        self.gradient_weights_n = grad_weights_hidden_acc
        # We don't expose output grads via property, but we use them for update below

        # --- Optimizer Update ---
        if self.optimizer:
            self.fc_hidden.weights = self.optimizer.calculate_update(self.fc_hidden.weights, grad_weights_hidden_acc)
            self.fc_output.weights = self.optimizer.calculate_update(self.fc_output.weights, grad_weights_output_acc)

        if grad_input.shape[0] == 1:
            return grad_input[0, :, :]
        return grad_input

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    def gradient_weights(self):
        return self.gradient_weights_n
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_weights_n = value

    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, value):
        self._memorize = value