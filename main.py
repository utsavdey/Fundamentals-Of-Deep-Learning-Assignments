from activation import *
from grad import *
import copy

last = 2
network = []
gradient = []


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
        gradient[i]['h'] = h_grad(network=network, gradient=gradient, layer=i)
        gradient[i]['a'] = a_grad(network=network, gradient=gradient, layer=i)
    for i in range(n - 1, 0, -1):
        gradient[i]['weight'] = w_grad(network=network, gradient=gradient, layer=i, x=x)
        gradient[i]['bias'] = gradient[i]['a']
    # back propagation ends


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
