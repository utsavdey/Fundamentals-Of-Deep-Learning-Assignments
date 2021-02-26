print('Hello World')
import numpy as np
import math
import copy

last = 2
network = []
gradient = []
input = []


def output_grad(y_hat, label):  # grad w.r.t out actiovation
    temp = np.zeros_like(y_hat)
    temp[label] = -1 / y_hat[label]
    return temp


def last_grad(y_hat, label):  # grad w.r.t out last layer
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    return temp


def a_grad(layer):  # grad w.r.t out a_i's layer
    active_grad_ = np.multiply(network[layer]['h'], 1 - network[layer]['h'])
    return np.multiply(gradient[layer]['h'],active_grad_)  # hadamard multiplication


def h_grad(layer):  # grad w.r.t out h_i layer
    network[layer]['weight'].transpose()
    return network[layer + 1]['weight'].transpose() @ gradient[layer + 1]['a']


def w_grad(a_grad_, h_k_1):
    return a_grad_ @ h_k_1.transpose()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    x = np.exp(x)
    x = x / np.sum(x)
    return x


def activation_function(a, _activate_func_callback):
    if _activate_func_callback == softmax:
        return softmax(a)
    g = np.empty_like(a)
    for i, elem in np.ndenumerate(a):
        g[i] = _activate_func_callback(elem)
    return g


def train(x, y):
    n = len(network)
    # forward propagation
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        if i == n - 1:
            network[i]['h'] = activation_function(network[i]['a'], softmax)  # last layer
        else:
            network[i]['h'] = activation_function(network[i]['a'], sigmoid)
    loss = -1 * np.log(network[n - 1]['h'][y])
    # forward propagation ends
    print(loss)
    # backpropagation starts
    gradient[n - 1]['h'] = output_grad(network[n - 1]['h'], y)
    gradient[n - 1]['a'] = last_grad(network[n - 1]['a'], y)
    for i in range(n - 2, 0, -1):
        gradient[i]['h'] = h_grad(layer=i)
        gradient[i]['a'] = a_grad(layer=i)


def master(layers, neuron_each_layer, k, x, y):
    n = neuron_each_layer
    for i in range(layers):  # making basic structure
        layer = {}
        if i == layers - 1:
            n = k
            layer['weight'] = np.random.rand(n, neuron_each_layer)
        else:
            layer['weight'] = np.random.rand(n, n)
        layer['bias'] = np.random.rand(n, 1)
        layer['h'] = np.ones((n, 1))
        layer['a'] = np.ones((n, 1))
        network.append(layer)
    global gradient
    gradient = copy.deepcopy(network)  # structure copy
    train(x=x, y=y)


master(layers=2, neuron_each_layer=4, k=2, x=np.array([1, .1, .4, .5]).reshape(4, 1), y=1)

print(activation_function(a=np.array([[.1, .2], [.3, .30]]), _activate_func_callback=sigmoid))
