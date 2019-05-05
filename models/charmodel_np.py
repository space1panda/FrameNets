import numpy as np
from utils.utils_np import clip

class CharModelRNN:

    def __init__(self, clip_ratio, params, seq_len):

        """Initialize necessary hyperparams and nn layers. Import updated params
        (or zero params).
        Reset all grads"""

        self._clip_ratio = clip_ratio
        self._Wax = params['Wax']
        self._Waa = params['Waa']
        self._Wya = params['Wya']
        self._b = params['b']
        self._by = params['by']
        self._seq_len = seq_len

    def forward_rnn(self, X, Y, hidden_back):

        hidden, output = {}, {}

        hidden[-1] = hidden_back
        loss = 0

        for t in range(self._seq_len):
            hidden[t] = np.tanh(np.dot(self._Wax, X[:, t].T) +
                                np.dot(self._Waa, hidden[t - 1]) + self._b)
            preoutput = np.dot(self._Wya, hidden[t]) + self._by
            output[t] = np.exp(preoutput) / np.sum(np.exp(preoutput))

            loss -= np.sum(np.log(output[t].T[range(len(X)), Y[:, t]])) / len(X)

        cache = (output, hidden)

        return loss, cache

    def rnn_backward(self, X, Y, cache):

        gradients = {}
        output, hidden = cache
        gradients['dWax'] = np.zeros_like(self._Wax)
        gradients['dWaa'] = np.zeros_like(self._Waa)
        gradients['dWya'] = np.zeros_like(self._Wya)
        gradients['db'] = np.zeros_like(self._b)
        gradients['dby'] = np.zeros_like(self._by)
        gradients['da_next'] = np.zeros_like(hidden[0])

        for t in reversed(range(self._seq_len)):
            dy = np.copy(output[t])
            dy[Y[:, t]] -= 1
            gradients['dWya'] += np.dot(dy, hidden[t].T)
            gradients['dby'] += np.sum(dy, axis=1).reshape(-1, 1)
            da = np.dot(self._Wya.T, dy) + gradients['da_next']
            daraw = (1 - hidden[t] * hidden[t]) * da
            gradients['db'] += np.sum(daraw, axis=1).reshape(-1, 1)
            gradients['dWax'] += np.dot(daraw, X[:, t])  # .reshape(1,-1))
            gradients['dWaa'] += np.dot(daraw, hidden[t - 1].T)
            gradients['da_next'] = np.dot(self._Waa.T, daraw)

        return gradients, hidden[t - 1]

    def optimize(self, X, Y, hidden_back):

        loss, cache = self.forward_rnn(X, Y, hidden_back)
        gradients, hidden_back = self.rnn_backward(X, Y, cache)
        gradients = clip(gradients, self._clip_ratio)
        return loss, gradients, hidden_back


