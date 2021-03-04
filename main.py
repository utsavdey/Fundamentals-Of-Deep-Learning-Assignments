"""Implement Feed Forward neural network where the parameters are
   number of hidden layers and number of neurons in each hidden layer"""

import copy
from keras.datasets import fashion_mnist
from grad import *
from activation import *

""" get training and testing vectors
    Number of Training Images = 60000
    Number of Testing Images = 10000 """
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

last = 2
# network is a list of all the learning parameters in every layer and gradient is its copy
network = []
gradient = []
# store gradient w.r.t a single datapoint
transient_gradient = []


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


def backward_propagation(number_of_layers, x, y, number_of_datapoint, clean=False):
    transient_gradient[number_of_layers - 1]['h'] = output_grad(network[number_of_layers - 1]['h'], y)
    transient_gradient[number_of_layers - 1]['a'] = last_grad(network[number_of_layers - 1]['a'], y)
    for i in range(number_of_layers - 2, -1, -1):
        transient_gradient[i]['h'] = h_grad(network=network, transient_gradient=transient_gradient, layer=i)
        transient_gradient[i]['a'] = a_grad(network=network, transient_gradient=transient_gradient, layer=i)
    for i in range(number_of_layers - 1, -1, -1):
        transient_gradient[i]['weight'] = w_grad(network=network, transient_gradient=transient_gradient, layer=i, x=x)
        transient_gradient[i]['bias'] = gradient[i]['a']
    if clean:
        gradient[number_of_layers - 1]['h'] = transient_gradient[number_of_layers - 1]['h'] / float(number_of_datapoint)
        gradient[number_of_layers - 1]['a'] = transient_gradient[number_of_layers - 1]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            gradient[i]['h'] = transient_gradient[i]['h'] / float(number_of_datapoint)
            gradient[i]['a'] = transient_gradient[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            gradient[i]['weight'] = transient_gradient[i]['weight'] / float(number_of_datapoint)
            gradient[i]['bias'] = transient_gradient[i]['bias'] / float(number_of_datapoint)
    else:

        gradient[number_of_layers - 1]['h'] += transient_gradient[number_of_layers - 1]['h'] / float(
            number_of_datapoint)
        gradient[number_of_layers - 1]['a'] += transient_gradient[number_of_layers - 1]['a'] / float(
            number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            gradient[i]['h'] += transient_gradient[i]['h'] / float(number_of_datapoint)
            gradient[i]['a'] += transient_gradient[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            gradient[i]['weight'] += transient_gradient[i]['weight'] / float(number_of_datapoint)
            gradient[i]['bias'] += transient_gradient[i]['bias'] / float(number_of_datapoint)


def descent(eta, layers, number_of_data_points):
    for i in range(layers):
        network[i]['weight'] -= (eta / float(number_of_data_points)) * gradient[i]['weight']
        network[i]['bias'] -= (eta / float(number_of_data_points)) * gradient[i]['bias']


# 1 epoch = 1 pass over the data
def train(datapoints, epochs, labels, f):
    n = len(network)  # number of layers
    # f = len(datapoints[0])  # number of features
    d = len(datapoints)  # number of data points
    # forward propagation
    for i in range(epochs):
        clean = True
        for j in range(d):
            # creating a single data vector and normalising color values between 0 to 1
            x = datapoints[j].reshape(784, 1) / 255.0
            y = labels[j]
            forward_propagation(n, x)
            backward_propagation(n, x, y, number_of_datapoint=d, clean=clean)
            clean = False

        descent(eta=.01, layers=n, number_of_data_points=d)
        loss = -1 * np.log(network[n - 1]['h'][y])

        # forward propagation ends


"""master() is used to intialise all the learning parameters 
   in every layer and then start the training process"""


def master(layers, neurons_in_each_layer, epochs, k, x, y):
    n = neurons_in_each_layer

    """intializing number of input features per datapoint as 784, 
       since dataset consists of 28x28 pixel grayscale images """
    n_features = 784

    for i in range(layers):
        # Initialize an Empty Dictionary: layer
        layer = {}
        # Weight matrix depends on number of features in the first layer
        if i == 0:
            layer['weight'] = np.random.normal(size=(n, n_features))
            glorot = n_features
        elif i == layers - 1:
            # special handling for the last layer.
            n = k
            """Create an array of size [number of classes * neurons last hidden layer] and fill it with random values
               from a Gaussian Distribution having 0 mean and 1 S.D."""
            layer['weight'] = np.random.normal(size=(n, neurons_in_each_layer))
            glorot = neurons_in_each_layer
        else:
            # Assuming the number of neurons in every hidden layer is the same
            layer['weight'] = np.random.normal(size=(n, n))
            glorot = n
        # glorot inittialization. Vanishing and exploding gradient.
        layer['weight'] = layer['weight'] * math.sqrt(1 / float(glorot))
        # initialise a 1-D array of size n with random samples from a uniform distribution over [0, 1).
        layer['bias'] = np.random.rand(n, 1)
        # initialises a 2-D array of size [n*1] and type float with element having value as 1.
        layer['h'] = np.ones((n, 1))
        layer['a'] = np.ones((n, 1))
        network.append(layer)
    global gradient
    """Recursively make a copy of network. Changes made to the copy will not reflect in the original network."""
    gradient = copy.deepcopy(network)
    global transient_gradient
    transient_gradient = copy.deepcopy(network)
    train(datapoints=trainX, labels=trainy, epochs=epochs, f=n_features)


master(layers=3, neurons_in_each_layer=3, epochs=2, k=10, x=trainX[1:3], y=trainy[1:3])

