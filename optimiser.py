import sys
import copy
import math

"""This file contains various gradient optimisers"""


# class for simple gradient descent
class SimpleGradientDescent:
    def __init__(self, eta, layers):
        # learning rate
        self.eta = eta
        # historical loss, will be required for rate annealing
        self.hist_loss = sys.float_info.max
        # number of layers
        self.layers = layers

    # function for gradient descending
    def descent(self, network, gradient):
        for i in range(self.layers):
            network[i]['weight'] -= self.eta * gradient[i]['weight']
            network[i]['bias'] -= self.eta * gradient[i]['bias']

    # function for learning rate annealing
    def anneal(self, loss):
        # if loss increases decrease learning rate
        if loss > self.hist_loss:
            self.eta = self.eta / 2.0
        self.hist_loss = loss


# class for simple gradient descent
class MomentumGradientDescent:
    def __init__(self, eta, layers, gamma):
        # learning rate
        self.eta = eta
        self.gamma = gamma
        # historical loss, will be required for rate annealing
        self.hist_loss = sys.float_info.max
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # historical momentum
        self.momentum = None

    # function for gradient descending
    def descent(self, network, gradient):
        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        if self.momentum == None:
            # copy the stucture
            self.momentum = copy.deepcopy(gradient)
            # initialize momenutum
            for i in range(self.layers):
                self.momentum[i]['weight'] = (1 - gamma) * gradient[i]['weight']
                self.momentum[i]['bias'] = (1 - gamma) * gradient[i]['bias']
        else:
            # update momentum
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + (1 - gamma) * gradient[i]['weight']
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + (1 - gamma) * gradient[i]['bias']
        # the descent
        for i in range(self.layers):
            network[i]['weight'] -= self.eta * self.momentum[i]['weight']
            network[i]['bias'] -= self.eta * self.momentum[i]['bias']

        self.calls += 1

    # function for learning rate annealing
    def anneal(self, loss):
        # if loss increases decrease learning rate
        if loss > self.hist_loss:
            self.eta = self.eta / 2.0
        self.hist_loss = loss