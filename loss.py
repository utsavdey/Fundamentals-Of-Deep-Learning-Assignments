"""This file will contain various methods for calculation of loss functions"""
import numpy as np


# calculate cross entropy
def cross_entropy(label, softmax_output):
    # as we have only one true label, we have simplified the function for faster calculation.
    return -np.log(softmax_output[label])
