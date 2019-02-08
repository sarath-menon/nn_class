import numpy as np
import h5py
import sys


def sigmoid(x):
    g = 1.0 / (1.0 + np.exp(-x))
    return g

def tanh(x):
    g = np.tanh(x)
    return g

def Relu(x):
    g = np.maximum(0 ,x)
    return g

def tanh_backward(A1 ,dZ2):
    d_gz = 1 - np.power(A1, 2)
    dZ1 = np.dot(W2.T, dZ2) * d_gz # nh*m  multiply by local gradient
    return dZ1

def sigmoid_backward(x):
    dg = np.dot(x ,1-x)
    return dg

def Relu_backward(dA1 ,Z1):
    dZ1 = np.array(dA1 ,copy=True) # Relu backward
    dZ1[Z1 <= 0] = 0
    return dZ1

def Optimizer(optimizer ,parameters ,grad ,learning_rate  ,v_grad ,beta=0.9):
    for l in range(len(parameters)-2):
        if (optimizer == 'Momentum'):
            v_grad["v_dW" + str(l+1)] = (beta * v_grad["v_dW" + str(l+1)]) +  0.9*grad["dW" + str(l+1)]
            v_grad["v_db" + str(l+1)] = (beta * v_grad["v_db" + str(l+1)]) +  0.9*grad["db" + str(l+1)]

            parameters["W" + str(l+1)] -= learning_rate * v_grad["v_dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * v_grad["v_db" + str(l+1)]

        if (optimizer == 'RMS_prop'):
            v_grad["v_dW" + str(l+1)] = (beta * v_grad["v_dW" + str(l+1)]) +  0.9*np.square(grad["dW" + str(l+1)])
            v_grad["v_db" + str(l+1)] = (beta * v_grad["v_db" + str(l+1)]) +  0.9*np.square(grad["db" + str(l+1)])

            parameters["W" + str(l+1)] -= learning_rate * (grad["dW" + str(l+1)] / (np.sqrt(v_grad["v_dW" + str(l+1)])+1e-1))
            parameters["b" + str(l+1)] -= learning_rate * (grad["db" + str(l+1)] / (np.sqrt(v_grad["v_db" + str(l+1)])+1e-8))

        if (optimizer == 'Gradient_descent'):
            parameters["W" + str(l+1)] -= learning_rate * grad["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grad["db" + str(l+1)]

    return parameters


def backprop_layer(parameters ,dZ ,cache ,grad ,layer ,m ,activation_function):
    dW = np.dot(dZ, cache['A'+str(layer-1)].T) / m  # (ny*nh)
    db = np.sum(dZ, axis=1, keepdims=True) / m  # ny*1
    dA = np.dot(parameters['W'+str(layer)].T, dZ)
    if activation_function == tanh:dZ_prev = tanh_backward(cache['A'+str(layer-1)] ,dZ3)
    if activation_function == Relu:dZ_prev = Relu_backward(dA ,cache['Z'+str(layer-1)])

    grad['dW'+str(layer)] = dW
    grad['db'+str(layer)] = db
    return dZ_prev ,grad

def load_data():
    train_dataset = h5py.File(sys.path[0]+'/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(sys.path[0]+'/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
