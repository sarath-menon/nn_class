import numpy as np
from helper_functions import *
import random
import matplotlib.pyplot as plt
from nn_class import nn
random.seed(1)

class reinforce(nn):
    '''Basic REINFORCE Algorithm'''
    def __init__(self ,X_shape ,Y_shape):
        super().__init__(X_shape,Y_shape)
        pass

    def calc_discounted_reward(reward_book):
        R = 0
        for r in reversed(reward_book):
            R = r + discount_factor * R
            rewards.insert(0, R)
        return rewards

    def select_action(obs):
        print('obs:{}'.format(obs))
        obs.shape = (4,1)
        action = net.predict(obs)
        return action

    def random_sampling(action_space):
        action = np.random.randint(action_space)
        return action

    def Backprop(self ,reward_multiplier):
        L = len(self.hidden_layers)
        self.cache['A0'] = self.X_train
        if (self.activation_functions[L] == sigmoid or self.activation_functions[L] ==softmax):
            grad = (self.y_pred - self.Y_train) * reward_multiplier  # (n_y*m)

        for l in range(L+1):
            self.grads['dW'+str(L-l+1)] = np.dot(grad, self.cache['A'+str(L-l)].T) / self.m  # (ny*nh)
            self.grads['db'+str(L-l+1)] = np.sum(grad, axis=1, keepdims=True) / self.m  # ny*1
            if self.activation_functions[l] == Relu:
                dA = np.dot(self.parameters['W'+str(L-l+1)].T, grad)
                grad = Relu_backward(dA ,self.cache['Z'+str(L-l)])
