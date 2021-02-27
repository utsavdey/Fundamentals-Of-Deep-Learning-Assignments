from activation import *
from grad import *
import copy

last = 2
network = []
gradient = []


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        if i == n - 1:
            network[i]['h'] = activation_function(network[i]['a'], softmax)  # last layer
        else:
            network[i]['h'] = activation_function(network[i]['a'], sigmoid)


def backward_propagation(n, x, y, clean=False):
    if clean:
        gradient[n - 1]['h'] = output_grad(network[n - 1]['h'], y)
        gradient[n - 1]['a'] = last_grad(network[n - 1]['a'], y)
        for i in range(n - 2, 0, -1):
            gradient[i]['h'] = h_grad(network=network, gradient=gradient, layer=i)
            gradient[i]['a'] = a_grad(network=network, gradient=gradient, layer=i)
        for i in range(n - 1, 0, -1):
            gradient[i]['weight'] = w_grad(network=network, gradient=gradient, layer=i, x=x)
            gradient[i]['bias'] = gradient[i]['a']
    else:

        gradient[n - 1]['h'] += output_grad(network[n - 1]['h'], y)
        gradient[n - 1]['a'] += last_grad(network[n - 1]['a'], y)
        for i in range(n - 2, 0, -1):
            gradient[i]['h'] += h_grad(network=network, gradient=gradient, layer=i)
            gradient[i]['a'] += a_grad(network=network, gradient=gradient, layer=i)
        for i in range(n - 1, 0, -1):
            gradient[i]['weight'] += w_grad(network=network, gradient=gradient, layer=i, x=x)
            gradient[i]['bias'] += gradient[i]['a']


def descent(eta, layers, number_of_data_points):
    for i in range(layers):
        network[i]['weight'] -= (eta / float(number_of_data_points)) * gradient[i]['weight']
        network[i]['bias'] -= (eta / float(number_of_data_points)) * gradient[i]['bias']


def train(datapoints, epochs, labels):
    n = len(network)  # number of layers
    f = len(datapoints[0])  # number of features
    d = len(datapoints)  # number of data points
    # forward propagation
    for i in range(epochs):
        clean = True
        for (data, label) in zip(datapoints, labels):
            x = np.array(data).reshape(f, 1)  # creating a single data vector
            y = label
            forward_propagation(n, x)

            # backpropagation starts
            backward_propagation(n, x, y, clean=clean)
            clean = False
            descent(eta=.6, layers=n, number_of_data_points=d)
        loss = -1 * np.log(network[n - 1]['h'][y])
        # forward propagation ends
        print(loss)  # w.r.t one point only



def master(layers, neuron_each_layer, k, x, y):
    n = neuron_each_layer
    n_features = len(x[0])
    for i in range(layers):  # making basic structure
        layer = {}
        if i == 0:  # Weight matrix depends on number of features in the first layer
            layer['weight'] = np.random.rand(n, n_features)
        elif i == layers - 1:  # special handling for the last layer.
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
    train(datapoints=x, labels=y, epochs=2)


master(layers=3, neuron_each_layer=4, k=2, x=[[.1, .1, .4], [1, 1, .6]], y=[1, 0])

print(activation_function(a=np.array([[.1, .2], [.3, .30]]), _activate_func_callback=sigmoid))
