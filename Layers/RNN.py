import numpy as np
import Base

class RNN(Base.BaseLayer):
    """
    Recurrent Neural Network Layer
    """

    def __init__(self, input_size, hidden_size, output_size, optimizer=None):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer

        self.weights_ih = None  # Weights for input to hidden
        self.weights_hh = None  # Weights for hidden to hidden
        self.bias_h = None      # Bias for hidden layer
        self.last_hidden_state = np.zeros((1, hidden_size))
        self.cache = None
        self._memorize = False
        self._regularizer = None
        self._weights_initializer = None
        self._bias_initializer = None

    def initialize(self, weights_initializer=None, bias_initializer=None):
        # Use custom initializers if provided, else default
        if weights_initializer is not None:
            self._weights_initializer = weights_initializer
        if bias_initializer is not None:
            self._bias_initializer = bias_initializer
        if self._weights_initializer is not None:
            self.weights_ih = self._weights_initializer.initialize((self.input_size, self.hidden_size), self.input_size, self.hidden_size)
            self.weights_hh = self._weights_initializer.initialize((self.hidden_size, self.hidden_size), self.hidden_size, self.hidden_size)
        else:
            limit_ih = np.sqrt(1 / self.input_size)
            limit_hh = np.sqrt(1 / self.hidden_size)
            self.weights_ih = np.random.uniform(-limit_ih, limit_ih, (self.input_size, self.hidden_size))
            self.weights_hh = np.random.uniform(-limit_hh, limit_hh, (self.hidden_size, self.hidden_size))
        if self._bias_initializer is not None:
            self.bias_h = self._bias_initializer.initialize(self.hidden_size, 1, self.hidden_size)
        else:
            self.bias_h = np.zeros(self.hidden_size)

    def forward(self, input_tensor):
        batch_size, seq_length, _ = input_tensor.shape
        # If memorize is True, use last hidden state, else zeros
        if self._memorize and self.last_hidden_state.shape[0] == batch_size:
            h_t = self.last_hidden_state.copy()
        else:
            h_t = np.zeros((batch_size, self.hidden_size))
        self.cache = []

        for t in range(seq_length):
            x_t = input_tensor[:, t, :]
            h_t = np.tanh(np.dot(x_t, self.weights_ih) + np.dot(h_t, self.weights_hh) + self.bias_h)
            self.cache.append((x_t, h_t))

        self.last_hidden_state = h_t.copy()  # Save for next sequence if memorize
        return h_t

    def backward(self, error_tensor):
        batch_size, seq_length, _ = error_tensor.shape
        dW_ih = np.zeros_like(self.weights_ih)
        dW_hh = np.zeros_like(self.weights_hh)
        db_h = np.zeros_like(self.bias_h)
        dh_next = np.zeros((batch_size, self.hidden_size))

        for t in reversed(range(seq_length)):
            x_t, h_t = self.cache[t]
            dh = error_tensor[:, t, :] + dh_next
            dtanh = (1 - h_t ** 2) * dh

            dW_ih += np.dot(x_t.T, dtanh)
            # For dW_hh, use previous hidden state (h_{t-1})
            h_prev = self.cache[t-1][1] if t > 0 else np.zeros_like(h_t)
            dW_hh += np.dot(h_prev.T, dtanh)
            db_h += np.sum(dtanh, axis=0)
            dh_next = np.dot(dtanh, self.weights_hh.T)
        if self.optimizer:
            self.weights_ih = self.optimizer.update(self.weights_ih, dW_ih)
            self.weights_hh = self.optimizer.update(self.weights_hh, dW_hh)
            self.bias_h = self.optimizer.update(self.bias_h, db_h)
        # Return error tensor for previous layer (input)
        return dh_next
    
    @property
    def gradient_weights(self):
        # Stack weights for access: [weights_ih, weights_hh]
        return np.concatenate([
            self.weights_ih.flatten(),
            self.weights_hh.flatten()
        ])

    @gradient_weights.setter
    def gradient_weights(self, value):
        # Split and reshape value into weights_ih and weights_hh
        ih_size = self.input_size * self.hidden_size
        hh_size = self.hidden_size * self.hidden_size
        self.weights_ih = value[:ih_size].reshape(self.input_size, self.hidden_size)
        self.weights_hh = value[ih_size:ih_size+hh_size].reshape(self.hidden_size, self.hidden_size)
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    def calculate_regularization_loss(self):
        loss = 0
        if self._regularizer is not None:
            loss += self._regularizer.loss(self.weights_ih)
            loss += self._regularizer.loss(self.weights_hh)
        return loss

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value):
        self._regularizer = value
    
    @property
    def memorize(self):
        return self._memorize
    @memorize.setter
    def memorize(self, value):
        self._memorize = value