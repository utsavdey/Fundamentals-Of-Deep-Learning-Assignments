import numpy as np


def output_grad(y_hat, label):  # grad w.r.t out activation
    temp = np.zeros_like(y_hat)
    temp[label] = -1 / y_hat[label]
    return temp/np.linalg.norm(temp)


def last_grad(y_hat, label):  # grad w.r.t out last layer
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    return temp/np.linalg.norm(temp)


def a_grad(network, gradient, layer):  # grad w.r.t out a_i's layer
    active_grad_ = np.multiply(network[layer]['h'], 1 - network[layer]['h'])
    z=np.multiply(gradient[layer]['h'], active_grad_)
    z=z/np.linalg.norm(z)
    print(z)
    return z # hadamard multiplication


def h_grad(network, gradient, layer):  # grad w.r.t out h_i layer
    network[layer]['weight'].transpose()
    z=network[layer + 1]['weight'].transpose() @ gradient[layer + 1]['a']
    z=z/np.linalg.norm(z)
    return z


def w_grad(network, gradient, layer, x):
    if layer == 0:
        return gradient[layer]['a'] @ x.transpose()
    else:
        return gradient[layer]['a'] @ network[layer - 1]['h'].transpose()
