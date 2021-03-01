import numpy as np
import math
def sigmoid(x):
  #if-else to prevent math overflow
    if x>=0:
      return 1 / (1 + math.exp(-x))
    else:
      return math.exp(x) / (math.exp(x) + 1)

def softmax(x):
    x = np.exp(x)
    x = x / np.sum(x)
    return x


def activation_function(a, _activate_func_callback):
    if _activate_func_callback == softmax:
        return softmax(a)
    g = np.empty_like(a)
    for i, elem in np.ndenumerate(a):
        g[i] = _activate_func_callback(elem)
    return g