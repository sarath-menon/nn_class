import numpy as np
from helper_functions import *
import random
import matplotlib.pyplot as plt
random.seed(1)

class reinforce(nn):
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

    def random_sampling(action_space):
        action = np.random.randint(action_space)
        return action

    def gradient_ascent(self ,optimizer ,beta):
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
