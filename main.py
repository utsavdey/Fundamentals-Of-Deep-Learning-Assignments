"""Implement Feed Forward neural network where the parameters are
   number of hidden layers and number of neurons in each hidden layer"""

import copy
from keras.datasets import fashion_mnist
from loss import *
from grad import *
from activation import *
from optimiser import *
import wandb

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
# will contain the total amount of loss for each timestep(1). One timestep is defined as one update of the parameters.
loss = 0


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        network[i]['h'] = activation_function(network[i]['a'], context=network[i]['context'])


def backward_propagation(number_of_layers, x, y, number_of_datapoint, loss_type, clean=False):
    transient_gradient[number_of_layers - 1]['h'] = output_grad(network[number_of_layers - 1]['h'], y,
                                                                loss_type=loss_type)
    transient_gradient[number_of_layers - 1]['a'] = last_grad(network[number_of_layers - 1]['h'], y)
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


# this function is used for validation, useful during hyperparameter tuning or model change.
def validate(number_of_layer, validateX, validateY, loss_type):
    loss = 0
    acc = 0
    if loss_type == 'cross_entropy':
        for x, y in zip(validateX, validateY):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding loss w.r.t to a single datapoint
            loss += cross_entropy(label=y, softmax_output=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    elif loss_type == 'squared_error':
        for x, y in zip(validateX, validateY):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding loss w.r.t to a single datapoint
            loss += squared_error(label=y, softmax_output=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    average_loss = loss / float(len(validateX))
    acc = acc / float(len(validateX))
    return [average_loss, acc]


# 1 epoch = 1 pass over the data
def fit(datapoints, batch, epochs, labels, opt, f, learning_rate, loss_type):
    n = len(network)  # number of layers
    d = len(datapoints)  # number of data points
    """This variable will be used to separate , training and validation set
        1) we take 10 % of the data as suggested in the question. -->int(d * .1)
        2) we also add any extra remaining data to validation set so that,
        training data is exactly divisible by batch size -->((d - int(d * .1)) % batch
    """
    border = d - ((d - int(d * .1)) % batch + int(d * .1))
    # separating the validation data
    validateX = datapoints[border:]
    validateY = labels[border:]
    # deleting copied datapoints
    datapoints = datapoints[:border]
    labels = labels[:border]
    # updating d
    d = border
    # is used to stochastically select our data.
    shuffler = np.arange(0, d)
    # creating simple gradient descent optimiser

    # loop for epoch iteration
    for k in range(epochs):
        # iteration for different starting point for epoch
        # shuffler at the start of each epoch
        np.random.shuffle(shuffler)
        for i in range(0, d - batch + 1, batch):
            clean = True
            # initiating loss for current epoch
            global loss
            loss = 0
            if isinstance(opt, NAG):
                opt.lookahead(network=network)
            # iterate over a batch
            for j in range(i, i + batch, 1):
                # creating a single data vector and normalising color values between 0 to 1
                x = datapoints[shuffler[j]].reshape(784, 1) / 255.0
                y = labels[shuffler[j]]
                forward_propagation(n, x)

                backward_propagation(n, x, y, number_of_datapoint=batch, loss_type=loss_type, clean=clean)
                clean = False

            opt.descent(network=network, gradient=gradient)
            validation_result = validate(number_of_layer=n, validateX=validateX, validateY=validateY,
                                         loss_type=loss_type)
            print(validation_result, training_result)

        # for wandb logging
        validation_result = validate(number_of_layer=n, validateX=validateX, validateY=validateY,
                                     loss_type=loss_type)
        training_result = validate(number_of_layer=n, validateX=datapoints[i, i + batch],
                                   validateY=labels[i, i + batch], loss_type=loss_type)

        # printing average loss.
        wandb.log({"val_accuracy": validation_result[1], 'val_loss': validation_result[0][0],
                   'train_accuracy': training_result[1], 'train_loss': training_result[0][0], 'epoch': k})

        if np.isnan(validation_result[0])[0]:
            return


""" Adds a particular on top of previous layer , the layers are built in a incremental way.
    Context denotes the type of layer we have.Eg - Sigmoid or Tanh etc.
    Passing any number to input_dim it we counted as the first layer
 """


def add_layer(number_of_neurons, context, weight_init, input_dim=None):
    # Initialize an Empty Dictionary: layer
    layer = {}
    if weight_init == 'random':
        if input_dim is not None:
            layer['weight'] = np.random.rand(number_of_neurons, input_dim)
        else:
            # get number of neurons in the previous layer
            previous_lay_neuron_num = network[-1]['h'].shape[0]
            layer['weight'] = np.random.rand(number_of_neurons, previous_lay_neuron_num)

    elif weight_init == 'xavier':
        if input_dim is not None:
            layer['weight'] = np.random.normal(size=(number_of_neurons, input_dim))
            xavier = input_dim
        else:
            # get number of neurons in the previous layer
            previous_lay_neuron_num = network[-1]['h'].shape[0]
            layer['weight'] = np.random.normal(size=(number_of_neurons, previous_lay_neuron_num))
            xavier = previous_lay_neuron_num
        if context == 'relu':
            # relu has different optimal weight initialization.
            layer['weight'] = layer['weight'] * math.sqrt(2 / float(xavier))
        else:
            layer['weight'] = layer['weight'] * math.sqrt(1 / float(xavier))
    # initialise a 1-D array of size n with random samples from a uniform distribution over [0, 1).
    layer['bias'] = np.zeros((number_of_neurons, 1))
    # initialises a 2-D array of size [n*1] and type float with element having value as 1.
    layer['h'] = np.zeros((number_of_neurons, 1))
    layer['a'] = np.zeros((number_of_neurons, 1))
    layer['context'] = context
    network.append(layer)


"""master() is used to initialise all the learning parameters 
   in every layer and then start the training process"""


def master(layers, neurons_in_each_layer, batch, epochs, output_dim, x, y, learning_rate, activation,
           opt, weight_init='xavier'):
    n = neurons_in_each_layer

    """intializing number of input features per datapoint as 784, 
       since dataset consists of 28x28 pixel grayscale images """
    n_features = 784
    global network
    global gradient
    global transient_gradient
    network = []
    gradient = []
    transient_gradient = []
    # adding layers
    add_layer(number_of_neurons=32, context=activation, input_dim=784, weight_init=weight_init)
    # creating hidden layers
    add_layer(number_of_neurons=66, context=activation, weight_init=weight_init)
    add_layer(number_of_neurons=16, context=activation, weight_init=weight_init)
    add_layer(number_of_neurons=output_dim, context='softmax', weight_init=weight_init)

    """Copying the structure of network."""
    gradient = copy.deepcopy(network)
    transient_gradient = copy.deepcopy(network)
    fit(datapoints=trainX, labels=trainy, batch=batch, epochs=epochs, f=n_features, opt=opt,
        learning_rate=learning_rate, loss_type='cross_entropy')


def train():
    run = wandb.init()
    wandb.run.name = wandb.run.id
    if run.config.optimiser == 'nag':
        opti = NAG(layers=4, eta=run.config.learning_rate, gamma=.80, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'rmsprop':
        opti = RMSProp(layers=4, eta=run.config.learning_rate, beta=.80, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'sgd':
        opti = SimpleGradientDescent(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'mom':
        opti = MomentumGradientDescent(layers=4, eta=run.config.learning_rate, gamma=.99,
                                       weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'adam':
        opti = ADAM(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'nadam':
        opti = NADAM(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)

    master(layers=4, neurons_in_each_layer=8, epochs=run.config.epoch, batch=run.config.batch_size, output_dim=10,
           x=trainX,
           y=trainy, learning_rate=.0005,
           opt=opti, weight_init=run.config.weight_init, activation=run.config.activation)


wandb.agent(sweep_id='fwxtkhkk', function=train)
