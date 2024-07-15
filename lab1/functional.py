import math
import numpy as np

class Linear:

    def __init__(self, in_features, out_features, bias= True):
        
        self.in_features = in_features
        self.out_features = out_features

        # Reference from Pytorch https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # shape (out_features, in_features) of weight
        k = float(1) / self.in_features
        self.weight = np.random.uniform(-math.sqrt(k), math.sqrt(k), (self.out_features, self.in_features))

        self.bias = np.random.uniform(0, 0, (self.out_features, 1))     
        if bias:
            # shape (out_features) of bias
            self.bias = np.random.uniform(-math.sqrt(k), math.sqrt(k), (self.out_features, 1))     
        
    def __call__(self, x):
        return self.__forward__(x)

    def __forward__(self, input):

        self.a = input
        return np.transpose(np.matmul(self.weight, input.T) + self.bias)
        
    def __update__(self):
        pass

# Activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def ReLU(x):
    return np.where(x >= 0, x, 0)

def derivative_ReLU(x):
    return np.where(x >= 0, 1, 0)

def LeakyReLU(x, a= 0.01):
    return np.where(x >= 0, x, a * x)

def derivative_LeakyReLU(x, a= 0.01):
    return np.where(x >= 0, 1, a)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.multiply(x, x)


# for report discussion

def without(x):
    return x

def derivative_without(x):
    return 1