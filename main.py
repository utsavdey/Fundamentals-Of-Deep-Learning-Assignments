import copy
from keras.datasets import fashion_mnist
from grad import *
from activation import *
# get training and testing vectors
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
last = 2
network = []
gradient = []
t = 0


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['h'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['h'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

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
            gradient[i]['a'] = gradient[i]['a'] / np.linalg.norm(gradient[i]['a'])
        for i in range(n - 1, 0, -1):
            gradient[i]['weight'] += w_grad(network=network, gradient=gradient, layer=i, x=x)
            gradient[i]['bias'] += gradient[i]['a']


def descent(eta, layers, number_of_data_points):
    for i in range(layers):
        network[i]['weight'] -= (eta / float(number_of_data_points)) * gradient[i]['weight']
        network[i]['bias'] -= (eta / float(number_of_data_points)) * gradient[i]['bias']


def train(datapoints, epochs, labels, f):
    n = len(network)  # number of layers
    # f = len(datapoints[0])  # number of features
    d = len(datapoints)  # number of data points
    # forward propagation
    for i in range(epochs):
        clean = True
        for (data, label) in zip(datapoints, labels):
            x = testX[0].reshape(784, 1) / 255.0  # creating a single data vector
            y = label
            forward_propagation(n, x)

            # backpropagation starts
            backward_propagation(n, x, y, clean=clean)
            clean = False
            descent(eta=.00000000000001, layers=n, number_of_data_points=d)
        loss = -1 * np.log(network[n - 1]['h'][y])
        # forward propagation ends


def master(layers, neuron_each_layer, k, x, y):
    n = neuron_each_layer
    n_features = 784
    for i in range(layers):  # making basic structure
        layer = {}

        if i == 0:  # Weight matrix depends on number of features in the first layer
            layer['weight'] = np.random.normal(size=(n, n_features))
            glorot = n_features
        elif i == layers - 1:  # special handling for the last layer.
            n = k
            layer['weight'] = np.random.normal(size=(n, neuron_each_layer))
            glorot = neuron_each_layer
        else:
            layer['weight'] = np.random.normal(size=(n, n))
            glorot = n
        layer['weight'] = layer['weight'] * math.sqrt(
            1 / float(glorot))  # glorot inittialization. Vanishing and exploding gradient.
        layer['bias'] = np.random.rand(n, 1)
        layer['bias'] = layer['bias']
        layer['h'] = np.ones((n, 1))
        layer['a'] = np.ones((n, 1))
        network.append(layer)
    global gradient
    gradient = copy.deepcopy(network)  # structure copy
    train(datapoints=x, labels=y, epochs=2, f=n_features)


master(layers=3, neuron_each_layer=3, k=10, x=testX, y=trainy)
