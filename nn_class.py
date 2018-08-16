import numpy as np
from helper_functions import *
import random
# from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from dnn_app_utils_v2 import load_data
import matplotlib.pyplot as plt
random.seed(1)

class nn:
    '''Basic neural network'''
    def __init__(self ,X ,Y ,Plot_loss=True):
        self.hidden_layers = []
        self.activation_functions = []
        self.parameters = {}
        self.grads = {}
        self.v_grad = {}
        self.Plot_loss = Plot_loss
        if self.Plot_loss==True : self.loss_cache = []
        self.X_train = X
        self.Y_train = Y
        self.y_pred = None
        self.m = X.shape[1]
        print('Input size :{},Output size :{} ,No of examples :{}'.format(X.shape[0],Y.shape[0],X.shape[1]))

    def create_model(self):
        L = len(self.hidden_layers)
        self.parameters['W1'] =np.random.randn(self.hidden_layers[0] ,self.X_train.shape[0]) / np.sqrt(self.X_train.shape[0])
        self.parameters['b1'] =np.random.randn(self.hidden_layers[0],1) / np.sqrt(self.hidden_layers[0])

        for l in range(len(self.hidden_layers)-1):
            self.parameters['W'+str(l+2)] = np.random.randn(self.hidden_layers[l+1],self.hidden_layers[l]) / np.sqrt(self.hidden_layers[l])
            self.parameters['b'+str(l+2)] = np.random.randn(self.hidden_layers[l+1],1) / np.sqrt(self.hidden_layers[l+1])
        self.parameters['W'+str(L+1)] = np.random.randn(self.Y_train.shape[0],self.hidden_layers[L-1]) / np.sqrt(self.hidden_layers[L-1])
        self.parameters['b'+str(L+1)] = np.random.randn(self.Y_train.shape[0],1) / np.sqrt(self.Y_train.shape[0])
        for layer,weights in self.parameters.items():print(layer,':',weights.shape)

    def create_layer(self ,hidden_size,activation_function):
        self.activation_functions.append(activation_function)
        if hidden_size != 'Input':
            self.hidden_layers.append(hidden_size)

    def forward_prop(self):
        A = self.X_train
        self.cache = {}
        for l in range(len(self.hidden_layers)+1):
            Z = np.dot(self.parameters['W'+str(l+1)] ,A) + self.parameters['b'+str(l+1)]
            A = self.activation_functions[l](Z)

            self.cache['A'+str(l+1)] = A
            self.cache['Z'+str(l+1)] = Z
            # print(self.activation_functions[l] ,l)
        self.y_pred = A

    def loss_function(self):
        cost = - np.sum(self.Y_train * np.log(self.y_pred) + (1 - self.Y_train) * np.log(1 - self.y_pred)) / self.m
        if self.Plot_loss==True : self.loss_cache.append(cost)
        print('loss:{}'.format(cost))

    def Backprop(self):
        L = len(self.hidden_layers)
        self.cache['A0'] = self.X_train
        if self.activation_functions[L] == sigmoid : grad = self.y_pred - self.Y_train   # (n_y*m)

        for l in range(L+1):
            self.grads['dW'+str(L-l+1)] = np.dot(grad, self.cache['A'+str(L-l)].T) / self.m  # (ny*nh)
            self.grads['db'+str(L-l+1)] = np.sum(grad, axis=1, keepdims=True) / self.m  # ny*1
            if self.activation_functions[l] == Relu:
                dA = np.dot(self.parameters['W'+str(L-l+1)].T, grad)
                grad = Relu_backward(dA ,self.cache['Z'+str(L-l)])

    def Optimizer(self ,optimizer ,beta):
        for l in range(len(self.hidden_layers)+1):
            if (optimizer == 'Momentum'):
                if ("v_dW" + str(l+1)) in self.v_grad != None :
                    self.v_grad["v_dW" + str(l+1)] = (beta * self.v_grad["v_dW" + str(l+1)]) +  0.9*self.grads["dW" + str(l+1)]
                    self.v_grad["v_db" + str(l+1)] = (beta * self.v_grad["v_db" + str(l+1)]) +  0.9*self.grads["db" + str(l+1)]

                    self.parameters["W" + str(l+1)] -= self.learning_rate * self.v_grad["v_dW" + str(l+1)]
                    self.parameters["b" + str(l+1)] -= self.learning_rate * self.v_grad["v_db" + str(l+1)]

                else:self.v_grad["v_dW" + str(l+1)] ,self.v_grad["v_db" + str(l+1)] = 0 ,0

            if (optimizer == 'Gradient_descent'):
                self.parameters["W" + str(l+1)] -= self.learning_rate * self.grads["dW" + str(l+1)]
                self.parameters["b" + str(l+1)] -= self.learning_rate * self.grads["db" + str(l+1)]

    def plot_loss(self):
        plt.figure()
        plt.title('Cross entropy loss')
        plt.plot(self.loss_cache ,label=self.optimizer)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train(self ,num_iterations=500 ,learning_rate=0.01 ,optimizer='Momentum' ,beta=0.9):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        for i in range(num_iterations):
            self.forward_prop()
            self.loss_function()
            self.Backprop()
            self.Optimizer(optimizer ,beta)
        if self.Plot_loss==True : self.plot_loss()
