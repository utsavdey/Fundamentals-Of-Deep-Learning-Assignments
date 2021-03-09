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
        # number of calls
        self.calls = 1
        # learning rate controller
        self.lrc = 1.0

    # function for gradient descending
    def descent(self, network, gradient):
        for i in range(self.layers):
            network[i]['weight'] -= self.eta / self.lrc * gradient[i]['weight']
            network[i]['bias'] -= self.eta / self.lrc * gradient[i]['bias']
        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


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
        # rate learning controller
        self.lrc = 1
        # historical momentum
        self.momentum = None

    # function for gradient descending
    def descent(self, network, gradient):
        """http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , Slide 70"""
        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            # initialize momentum- refer above lecture slide 36
            for i in range(self.layers):
                self.momentum[i]['weight'] = self.eta * gradient[i]['weight']
                self.momentum[i]['bias'] = self.eta * gradient[i]['bias']
        else:
            # update momentum
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + self.eta * gradient[i]['weight']
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + self.eta * gradient[i]['bias']
        # the descent
        for i in range(self.layers):
            network[i]['weight'] -= self.momentum[i]['weight']
            network[i]['bias'] -= self.momentum[i]['bias']

        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


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
        # learning rate controller
        self.lrc = 1.0

    # function for lookahead. Call this before forward propagation.
    def lookahead(self, network):
        # case when no momentum has been generated yet.
        if self.momentum is None:
            pass
        else:
            # update the gradient using momentum
            for i in range(self.layers):
                network[i]['weight'] -= self.momentum[i]['weight']
                network[i]['bias'] -= self.momentum[i]['bias']

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
                self.momentum[i]['weight'] = self.eta * gradient[i]['weight']
                self.momentum[i]['bias'] = self.eta * gradient[i]['bias']
        else:
            # update momentum: http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , slide: 46
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + self.eta / self.lrc * gradient[i][
                    'weight']
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + self.eta / self.lrc * gradient[i]['bias']

        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


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
        # learning rate controller
        self.lrc = 1.0

    # function for gradient descending
    def descent(self, network, gradient):

        # generate update for the next time step
        if self.update is None:
            # copy the structure
            self.update = copy.deepcopy(gradient)
            # initialize update at time step 1 assuming that update at time step 0 is 0
            for i in range(self.layers):
                self.update[i]['weight'] = (1 - self.beta) * (gradient[i]['weight']) ** 2
                self.update[i]['bias'] = (1 - self.beta) * (gradient[i]['bias']) ** 2
        else:
            for i in range(self.layers):
                self.update[i]['weight'] = self.beta * self.update[i]['weight'] + (1 - self.beta) * (gradient[i][
                    'weight']) ** 2
                self.update[i]['bias'] = self.beta * self.update[i]['bias'] + (1 - self.beta) * (
                gradient[i]['bias']) ** 2
        # Now we use the update rule for RMSProp
        for i in range(self.layers):
            network[i]['weight'] = network[i]['weight'] - np.multiply(
                (self.eta / np.sqrt(self.update[i]['weight'] + self.epsilon)), gradient[i]['weight'])
            network[i]['bias'] = network[i]['bias'] - np.multiply(
                (self.eta / np.sqrt(self.update[i]['bias'] + self.epsilon)), gradient[i]['bias'])

        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


# class for ADAM: Reference: https://arxiv.org/pdf/1412.6980.pdf?source=post_page---------------------------
"""Using the previous gradients instead of the previous updates allows the algorithm to continue changing 
   direction even when the learning rate has annealed significantly toward the end of training, resulting 
   in more precise fine-grained convergence"""


class ADAM:
    def __init__(self, eta, layers, beta1=0.9, beta2=0.999, eps=1e-8):
        # learning rate
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # first moment vector m_t: defined as a decaying mean over the previous gradients
        self.momentum = None
        # second moment vector v_t
        self.second_momentum = None
        # epsilon
        self.eps = eps
        # historical loss, will be required for rate annealing
        self.hist_loss = sys.float_info.max

    # function for gradient descending
    def descent(self, network, gradient):

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            self.second_momentum = copy.deepcopy(gradient)
            # initialize momentums
            for i in range(self.layers):
                # first momentum initialization
                self.momentum[i]['weight'] = (1 - self.beta1) * gradient[i]['weight']
                self.momentum[i]['bias'] = (1 - self.beta1) * gradient[i]['bias']
                # second momentum initialization
                self.second_momentum[i]['weight'] = (1 - self.beta2) * np.power(gradient[i]['weight'], 2)
                self.second_momentum[i]['bias'] = (1 - self.beta2) * np.power(gradient[i]['bias'], 2)
        else:
            for i in range(self.layers):
                # Update biased first moment estimate: Moving average of gradients
                self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * gradient[i][
                    'weight']
                self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i]['bias'
                ]
                # Update biased second raw moment estimate: rate adjusting parameter update similar to RMSProp
                self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (
                            1 - self.beta2) * np.power(gradient[i][
                                                           'weight'], 2)
                self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (
                            1 - self.beta2) * np.power(gradient[i]['bias'
                                                       ], 2)
        # bias correction
        for i in range(self.layers):
            self.momentum[i]['weight'] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['weight']
            self.momentum[i]['bias'] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias']

            self.second_momentum[i]['weight'] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i][
                'weight']
            self.second_momentum[i]['bias'] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i]['bias']

        # the descent
        for i in range(self.layers):
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['weight'])
            # add epsilon to square root of temp
            temp_eps = temp + self.eps
            # inverse everything
            temp_inv = 1 / temp_eps
            # perform descent for weight
            network[i]['weight'] = network[i]['weight'] - self.eta * (
                np.multiply(temp_inv, self.momentum[i]['weight']))

            # now we do the same for bias
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['bias'])
            # add epsilon to square root of temp
            temp_eps = temp + self.eps
            # inverse everything
            temp_inv = 1 / temp_eps
            # perform descent for weight
            network[i]['bias'] -= self.eta * np.multiply(temp_inv, self.momentum[i]['bias'])

        self.calls += 1

    # function for learning rate annealing
    def anneal(self, loss):
        # if loss increases decrease learning rate
        if loss > self.hist_loss:
            self.eta = self.eta / 2.0
        self.hist_loss = loss


# Reference: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
class NADAM:
    def __init__(self, eta, layers, beta1=0.9, beta2=0.999, eps=1e-8):
        # learning rate
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # first moment vector m_t: defined as a decaying mean over the previous gradients
        self.momentum = None
        # second moment vector v_t
        self.second_momentum = None
        # epsilon
        self.eps = eps
        # historical loss, will be required for rate annealing
        self.hist_loss = sys.float_info.max

    # function for gradient descending: Algorithm 2 Page 3
    def descent(self, network, gradient):

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            self.second_momentum = copy.deepcopy(gradient)
            # initialize momentums
            for i in range(self.layers):
                # first momentum initialization
                self.momentum[i]['weight'] = (1 - self.beta1) * gradient[i]['weight']
                self.momentum[i]['bias'] = (1 - self.beta1) * gradient[i]['bias']
                # second momentum initialization
                self.second_momentum[i]['weight'] = (1 - self.beta2) * np.power(gradient[i]['weight'], 2)
                self.second_momentum[i]['bias'] = (1 - self.beta2) * np.power(gradient[i]['bias'], 2)
        else:
            for i in range(self.layers):
                # Update biased first moment estimate: Moving average of gradients
                self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * \
                                             gradient[i][
                                                 'weight']
                self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i][
                    'bias'
                ]
                # Update biased second raw moment estimate: rate adjusting parameter update similar to RMSProp
                self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (
                        1 - self.beta2) * np.power(gradient[i][
                                                       'weight'], 2)
                self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (
                        1 - self.beta2) * np.power(gradient[i]['bias'
                                                   ], 2)
        # bias correction
        for i in range(self.layers):
            self.momentum[i]['weight'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i][
                'weight'] + ((1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['weight']
            self.momentum[i]['bias'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias'] + (
                        (1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['bias']

            self.second_momentum[i]['weight'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * \
                                                self.second_momentum[i][
                                                    'weight']
            self.second_momentum[i]['bias'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i][
                'bias']

        # the descent
        for i in range(self.layers):
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['weight'] + self.eps)
            # inverse everything
            temp_inv = 1 / temp
            # perform descent for weight
            network[i]['weight'] = network[i]['weight'] - self.eta * (
                np.multiply(temp_inv, self.momentum[i]['weight']))

            # now we do the same for bias
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['bias']) + self.eps
            # inverse everything
            temp_inv = 1 / temp
            # perform descent for weight
            network[i]['bias'] -= self.eta * np.multiply(temp_inv, self.momentum[i]['bias'])

        self.calls += 1
        # function for learning rate annealing

    def anneal(self, loss):
        # if loss increases decrease learning rate
        if loss > self.hist_loss:
            self.eta = self.eta / 2.0
        self.hist_loss = loss
