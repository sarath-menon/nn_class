import numpy as np
from helper_functions import *
import random
import matplotlib.pyplot as plt
import cloudpickle as pickle
random.seed(1)

class nn:
    '''Basic neural network'''
    def __init__(self ,X_shape ,Y_shape ,Plot_loss=True):
        self.hidden_layers = []
        self.activation_functions = []
        self.parameters = {}
        self.grads = {}
        self.v_grad = {}
        self.Plot_loss = Plot_loss
        if self.Plot_loss==True : self.loss_cache = []
        self.X_train_shape = X_shape
        self.Y_train_shape = Y_shape
        self.y_pred = None
        print('Input size :{},Output size :{} '.format(self.X_train_shape,self.Y_train_shape))

    def create_model(self):
        L = len(self.hidden_layers)
        self.parameters['W1'] =np.random.randn(self.hidden_layers[0] ,self.X_train_shape) / np.sqrt(self.X_train_shape)
        self.parameters['b1'] =np.random.randn(self.hidden_layers[0],1) / np.sqrt(self.hidden_layers[0])

        for l in range(len(self.hidden_layers)-1):
            self.parameters['W'+str(l+2)] = np.random.randn(self.hidden_layers[l+1],self.hidden_layers[l]) / np.sqrt(self.hidden_layers[l])
            self.parameters['b'+str(l+2)] = np.random.randn(self.hidden_layers[l+1],1) / np.sqrt(self.hidden_layers[l+1])
        self.parameters['W'+str(L+1)] = np.random.randn(self.Y_train_shape,self.hidden_layers[L-1]) / np.sqrt(self.hidden_layers[L-1])
        self.parameters['b'+str(L+1)] = np.random.randn(self.Y_train_shape,1) / np.sqrt(self.Y_train_shape)
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
        # print('ashape',A.shape)
        self.y_pred = A

    def loss_function(self):
        # print(self.Y_train[:,0] ,self.y_pred[:,0],np.sum(self.y_pred[:,0]) )
        cost = - np.sum(self.Y_train * np.log(self.y_pred) + (1 - self.Y_train) * np.log(1 - self.y_pred)) /self.m
        if self.Plot_loss==True : self.loss_cache.append(cost)
        print('loss:{}'.format(cost))

    def Backprop(self):
        L = len(self.hidden_layers)
        self.cache['A0'] = self.X_train
        if (self.activation_functions[L] == sigmoid or self.activation_functions[L] ==softmax):
            grad = self.y_pred - self.Y_train   # (n_y*m)

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

    def train(self ,X , Y ,epochs=5 ,learning_rate=0.001 ,optimizer='Momentum' ,beta=0.9 ,batch_size=128):
        self.m = X.shape[1]
        print('Dataset size:{} '.format(self.m))
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        for i in range(epochs):
            for x in range(int(self.m/batch_size)+1):
                self.X_train = X[:, batch_size*(x):batch_size*(x+1)]
                self.Y_train = Y[:, batch_size*(x):batch_size*(x+1)]
                self.forward_prop()
                self.Backprop()
                self.Optimizer(optimizer ,beta)
                print('Completed iteration {} of {} and epoch {} of {}'.format(x+1,int(self.m/batch_size)+1,i+1,epochs))
            self.loss_function()
            pickle.dump(self.parameters, open('save.p', 'wb'))
            print('model saved')
        if self.Plot_loss==True : self.plot_loss()

        def train(self ,X , Y ,learning_rate=0.001 ,optimizer='Momentum' ,beta=0.9 ,batch_size=128):
            self.m = X.shape[1]
            print('Dataset size:{} '.format(self.m))
            self.learning_rate = learning_rate
            self.optimizer = optimizer
            for i in range(epochs):
                for x in range(int(self.m/batch_size)+1):
                    self.X_train = X[:, batch_size*(x):batch_size*(x+1)]
                    self.Y_train = Y[:, batch_size*(x):batch_size*(x+1)]
                    self.forward_prop()
                    self.Backprop()
                    self.Optimizer(optimizer ,beta)
                    print('Completed iteration {} of {} and epoch {} of {}'.format(x+1,int(self.m/batch_size)+1,i+1,epochs))
                self.loss_function()
                pickle.dump(self.parameters, open('save.p', 'wb'))
                print('model saved')
            if self.Plot_loss==True : self.plot_loss()

    def load_saved_model(self, filename):
        self.parameters = pickle.load(open(filename, 'rb'))

    def predict(self ,A):
        for l in range(len(self.hidden_layers)+1):
           Z = np.dot(self.parameters['W'+str(l+1)] ,A) + self.parameters['b'+str(l+1)]
           A = self.activation_functions[l](Z)
        return A
