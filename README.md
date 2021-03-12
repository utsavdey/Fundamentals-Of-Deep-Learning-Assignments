# Assignment 1: Neural Network model development
----------------------------------------------------
In this project we implement a feed forward neural network and use gradient descent (and its variants: momentum, NAG, RMSProp, ADAM, NADAM) with backpropagation for classifying images from the fashion-mnist data over 10 class labels. We use wandb for visualisation of data for training, model comparison and accuracy of a large number of experiments that we have performed to make meaningful inferences.
# Libraries Used:
1. We have used numpy for all the mathematical calculations in forward propagation, back propagation algorithm and loss (cross entropy and squared error) function computation.
2. Scikit learn library was used to generate the confusion matrix which was converted into a dataframe using pandas library. 
3. Seaborn and matplotlib libraries were used for plotting the confusion matrix.
4. Keras and tensorflow was used for getting the fashion mnist dataset.
5. Pickle was used to save the best neural network model obtained during training.
# Installations: #
1. We have used pip as the package manager. All the libraries we used above can be installed using the command: `pip install -r requirements.txt`
2. Steps to Add Virtual Environment in IDE like Pycharm: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
# How to USE? #
The entire project has been modularised using functions and classes to make it as scalable as possible for future developments and extensions.
To train a model the project makes a call to `master()` in `main.py` file. </br>
The parameters in master are the following <br />
arg1 : batch : Number of datapoints to be in one batch. For e.g. 16, 32, 64<br />
arg2 : epochs : Number of passes to be done over the entire data<br />
arg3 : output_dim : Number of classes in the classification dataset<br />
arg4 : activation : The type of activation function used. The possible values are anyone one from `sigmoid, tanh, relu` <br />
arg5 : opt : An object of variants of gradient descents. The objects can be of type `SimpleGradientDescent`, `MomentumGradientDescent`, `NAG`, `RMSProp`, `ADAM`, `NADAM`<br />
arg6 : layer_1 : Number of neurons in layer 1.<br />
arg7 : layer_2 : Number of neurons in layer 2.<br />
arg8 : layer_3 : Number of neurons in layer 3.<br />
arg9 : loss : The loss function to be used for the classification dataset. The loss functions currently supported are `squared_error` and `cross_entropy`. <br />
arg10 : weight_init : Possible values are `random`, `xavier` for random weight initialisation and Xavier weight initialisation respectively.<br />
## Object initialization of optimiser classes ##
As mentioned above opt is of type `SimpleGradientDescent`, `MomentumGradientDescent`, `NAG`, `RMSProp`, `ADAM`, `NADAM`. Here we look at how opt can be initialized with respect to different optimiser classes supported.</br>
1. opt = SimpleGradientDescent(eta=<enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>)
2. opt = MomentumGradientDescent(eta=<enter learning rate value>, layers = <enter number of layers in the network>, gamma = <enter the gamma value>, weight_decay= <enter the weight decay value: Default value 0>)
3. opt = NAG(eta=<enter learning rate value>, layers = <enter number of layers in the network>, gamma = <enter the gamma value>, weight_decay= <enter the weight decay value: Default value 0>)
4. opt = RMSProp(eta=<enter learning rate value>, layers = <enter number of layers in the network>, beta = <enter the beta value>, weight_decay= <enter the weight decay value: Default value 0>)
5. opt = ADAM(eta=<enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>, beta1 = <enter the beta value: Default value 0.9>, beta2 = <enter the beta value: Default value 0.999, eps = <enter the epsilon value: Default value 1e-8>>) 
6. opt = NADAM(eta=<enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>, beta1 = <enter the beta value: Default value 0.9>, beta2 = <enter the beta value: Default value 0.999, eps = <enter the epsilon value: Default value 1e-8>)</br> 
***Example***:</br> 
`master(batch=495, epochs=7, output_dim=10, activation='tanh', opt=ADAM(eta=0.003576466933615937,layers=4,weight_decay=0.31834976996809683), layer_1 = 32,layer_2 = 64 ,layer_3 = 16 , loss='squared_error',weight_init='xavier')`</br>
</br> Here we have initialized an opt object of type ADAM having learning rate as 0.003576466933615937 and weight decay set to 0.31834976996809683. We are going to train a neural network of 4 layers where layer 1 consists of 32 neurons, layer 2 consists of 64 neurons and layer 3 consists of 16 neurons. The loss function used is a squared entropy loss function and the weights have been initialized using Xavier initialisation.
## How to change the number of layers, the number of neurons in each layer and the type of activation function used in each layer? ##
The *add_layer()* in master() provides the flexibility to change the number of layers, the number of neurons in each layer and the type of activation function used in each layer. It takes number of neurons(= number_of_neurons), type of activation function needed(= context) in the layer, the input dimension of the datset(= input_dim) and performs either random or Xavier weight initialisation based on the value passed to weight_init. </br>
***Example***: 
</br>`add_layer(number_of_neurons=layer_1, context=activation, input_dim=784, weight_init=weight_init)` where `layer_1 = 32`, `activation = 'tanh'`, `weight_init='xavier'`
</br> Hence in order to change the configuration of the neural network we only need to chenge the arguments passed to two functions:
1. master()
2. add_layer() inside master()
# Acknowledgements #
1. The entire project has been developed from the leacture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. http://www.cs.cmu.edu/~arielpro/15381f16/c_slides/781f16-17.pdf 
3. https://arxiv.org/pdf/1711.05101.pdf
4. https://wandb.ai
5. https://github.com/
6. https://stats.stackexchange.com/questions/153605
7. https://cs231n.github.io/neural-networks-3/#annealing-the-learning-rate

