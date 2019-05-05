import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def initialize_parameters(hidden_size, vocab_size):
    Wax = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
    Waa = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    Wya = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
    b = np.zeros((hidden_size, 1))  # hidden bias
    by = np.zeros((vocab_size, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    return parameters


def update_parameters_lronly(parameters, gradients, lr):
    parameters['Wax'] -= lr * gradients['dWax']
    parameters['Waa'] -= lr * gradients['dWaa']
    parameters['Wya'] -= lr * gradients['dWya']
    parameters['b'] -= lr * gradients['db']
    parameters['by'] -= lr * gradients['dby']

    return parameters


def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1

    newline_character = char_to_ix['\n']

    while idx != newline_character:
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        p = y
        idx = np.random.choice(list(range(y.size)), p=p.ravel())
        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a

    return indices


def chunks(l, n):
    for i in range(0, len(l), n): yield l[i:i + n]


def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


