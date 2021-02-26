print('Hello World')
import numpy as np
import math

last = 2
network = []
input = []


def output_grad(y_hat, label):  # grad w.r.t out actiovation
    e = np.zeros(y_hat.shape[0])
    e[label] = 1 / y_hat[label]
    return -e


def last_grad(y_hat, label):  # grad w.r.t out last layer
    e = np.zeros(y_hat.shape[0])
    e[label] = 1 - y_hat[label]
    return -e


def a_grad(h_grad_, active_grad_):  # grad w.r.t out a_i's layer
    return h_grad_.multiply(active_grad_)  # hadamard multiplication


def h_grad(w, a_grad_):  # grad w.r.t out h_i layer
    return w.transpose() @ a_grad_


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


def master(layers, neuron_each_layer, k, x, y):
    n = neuron_each_layer
    for i in range(layers):
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
    train(x=x, y=y)


master(layers=2, neuron_each_layer=4, k=2, x=np.array([1, .1, .4, .5]).reshape(4, 1), y=1)

print(activation_function(a=np.array([[.1, .2], [.3, .30]]), _activate_func_callback=sigmoid))
