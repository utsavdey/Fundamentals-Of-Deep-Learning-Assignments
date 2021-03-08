import sys
import copy
import math
import numpy as np

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


# class for Momentum gradient descent
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
        """http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , Slide 70"""
        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            # initialize momentum
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


# class for NAG
class NAG:
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

    # function for lookahead. Call this before forward propagation.
    def lookahead(self, network):
        # case when no momentum has been generated yet.
        if self.momentum is None:
            pass
        else:
            # update the gradient using momentum
            for i in range(self.layers):
                network[i]['weight'] -= self.eta * self.momentum[i]['weight']
                network[i]['bias'] -= self.eta * self.momentum[i]['bias']

    # function for gradient descending
    def descent(self, network, gradient):

        # the descent
        for i in range(self.layers):
            network[i]['weight'] -= self.eta * gradient[i]['weight']
            network[i]['bias'] -= self.eta * gradient[i]['bias']

        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        # generate momentum for the next time step next

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            # initialize momentum
            for i in range(self.layers):
                self.momentum[i]['weight'] = (1 - gamma) * gradient[i]['weight']
                self.momentum[i]['bias'] = (1 - gamma) * gradient[i]['bias']
        else:
            # update momentum: http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , slide: 46
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + (1 - gamma) * gradient[i][
                    'weight']
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + (1 - gamma) * gradient[i]['bias']

        self.calls += 1

    # function for learning rate annealing
    def anneal(self, loss):
        # if loss increases decrease learning rate
        if loss > self.hist_loss:
            self.eta = self.eta / 2.0
        self.hist_loss = loss

class RMSProp:
    def __init__(self, eta, layers, beta):
        # learning rate
        self.eta = eta
        # decay parameter for denominator
        self.beta = beta
        # historical loss, will be required for rate annealing
        self.hist_loss = sys.float_info.max
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # epsilon
        self.epsilon = 0.001
        # to implement update rule for RMSProp
        self.update = None

    # function for gradient descending
    def descent(self, network, gradient):


            # generate update for the next time step
            if self.update is None:
                # copy the structure
                self.update = copy.deepcopy(gradient)
                # initialize update at time step 1 assuming that update at time step 0 is 0
                for i in range(self.layers):
                    self.update[i]['weight'] = (1 - self.beta) * (gradient[i]['weight'])**2
                    self.update[i]['bias'] = (1 - self.beta) * (gradient[i]['bias'])**2
            else:
                for i in range(self.layers):
                    self.update[i]['weight'] = self.beta * self.update[i]['weight'] + (1 - self.beta) * (gradient[i][
                        'weight'])**2
                    self.update[i]['bias'] = self.beta * self.update[i]['bias'] + (1 - self.beta) * (gradient[i]['bias'])**2
            # Now we use the update rule for RMSProp
            for i in range(self.layers):
                network[i]['weight'] = network[i]['weight']-np.multiply((self.eta / np.sqrt(self.update[i]['weight']+self.epsilon)) , gradient[i]['weight'])
                network[i]['bias'] = network[i]['bias']-np.multiply((self.eta / np.sqrt(self.update[i]['bias']+self.epsilon)) , gradient[i]['bias'])

            self.calls += 1

    # function for learning rate annealing
    def anneal(self, loss):
            # if loss increases decrease learning rate
            if loss > self.hist_loss:
                self.eta = self.eta / 2.0
            self.hist_loss = loss