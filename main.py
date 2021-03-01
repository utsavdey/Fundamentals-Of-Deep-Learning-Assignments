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
network = []
gradient = []
t = 0


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['h'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['h'] = network[i]['weight'] @ network[i - 1]['a'] + network[i]['bias']

        if i == n - 1:
            print(network[i]['a'])
            network[i]['h'] = activation_function(network[i]['a'], softmax)  # last layer
            print(network[i]['h'])
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

# 1 epoch = 1 pass over the data
def train(datapoints, epochs, labels, f):
    n = len(network)  # number of layers
    # f = len(datapoints[0])  # number of features
    d = len(datapoints)  # number of data points
    # forward propagation
    print(datapoints[0].shape)
    for i in range(epochs):
        clean = True
        for i in range(d):
            x = datapoints[i].reshape(784, 1) / 255.0  # creating a single data vector
            y = labels[i]
            forward_propagation(n, x)
            clean = False
            # backpropagation starts
        backward_propagation(n, x, y, clean=clean)
        descent(eta=.01, layers=n, number_of_data_points=d)
        print(network)
        loss = -1 * np.log(network[n - 1]['h'][y])

        # forward propagation ends


def master(layers, neurons_in_each_layer, epochs, k, x, y):
    n = neurons_in_each_layer

    """intializing number of input features per datapoint as 784, 
           as dataset consists of 28x28 pixel grayscale images """
    n_features = 784

    for i in range(layers):  # making basic structure

        layer = {} #Initialize an Empty Dictionary: layer

        if i == 0:  # Weight matrix depends on number of features in the first layer
            layer['weight'] = np.random.normal(size=(n, n_features))
            glorot = n_features
        elif i == layers - 1:  # special handling for the last layer.
            n = k
            # Create an array of size [number of classes * neurons last hidden layer] and fill it with random values from a Gaussian Distribution having 0 mean and 1 S.D.
            layer['weight'] = np.random.normal(size=(n, neurons_in_each_layer))
            glorot = neurons_in_each_layer
        else:
            layer['weight'] = np.random.normal(size=(n, n)) # Assuming the number of neurons in every hidden layer is the same
            glorot = n
        layer['weight'] = layer['weight'] * math.sqrt(
            1 / float(glorot))  # glorot inittialization. Vanishing and exploding gradient.
        layer['bias'] = np.random.rand(n, 1) #initialise a 1-D array of size n with random samples from a uniform distribution over [0, 1).
        layer['h'] = np.ones((n, 1)) #initialises a 1-D array of size n and type float with element having valuue as 1.
        layer['a'] = np.ones((n, 1))
        network.append(layer)
    global gradient
    gradient = copy.deepcopy(network)  # structure copy. The changes made to a copy of network will not reflect in the original network.
    train(datapoints=trainX, labels=trainy, epochs=epochs, f=n_features)


master(layers=3, neurons_in_each_layer=3, epochs=6, k=10, x=trainX, y=trainy)

