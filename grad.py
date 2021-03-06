import numpy as np
import sys


def output_grad(y_hat, label):
    # grad w.r.t out activation
    temp = np.zeros_like(y_hat)
    temp[label] = -1 / y_hat[label]
    return temp


def last_grad(y_hat, label):
    # grad w.r.t out last layer
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    return temp


# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is sigmoid.
def sigmoid_grad(pre_activation_vector):
    return np.multiply(pre_activation_vector, 1 - pre_activation_vector)


# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is sigmoid.
def tanh_grad(pre_activation_vector):
    return 1-np.power(pre_activation_vector, 2)


def a_grad(network, transient_gradient, layer):
    # grad w.r.t  a_i's layer
    if network[layer]['context'] == 'sigmoid':
        active_grad_ = sigmoid_grad(network[layer]['a'])
    z = np.multiply(transient_gradient[layer]['h'], active_grad_)
    return z
    # hadamard multiplication


def h_grad(network, transient_gradient, layer):
    # grad w.r.t out h_i layer
    network[layer]['weight'].transpose()
    z = network[layer + 1]['weight'].transpose() @ transient_gradient[layer + 1]['a']
    return z


def w_grad(network, transient_gradient, layer, x):
    if layer == 0:
        return transient_gradient[layer]['a'] @ x.transpose()
    else:
        return transient_gradient[layer]['a'] @ network[layer - 1]['h'].transpose()
