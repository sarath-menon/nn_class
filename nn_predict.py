# # Dataset taken from Andrew NG's Deep learning course on coursera
from nn_class import nn
from helper_functions import *
import matplotlib.pyplot as plt
import random
import keras
from keras.datasets import mnist
random.seed(1)

(X_test, Y_test),(x_test, y_test) = mnist.load_data()

X_test = x_test.reshape(28*28 ,10000)

Y_test = keras.utils.to_categorical(y_test,10).T
print('Xshape',X_test.shape)
print('yshape',Y_test.shape)
del mnist

net = nn(X_test.shape[0] ,Y_test.shape[0] ,Plot_loss=False)
net.create_layer('Input',Relu)
net.create_layer(256 ,softmax)
net.load_saved_model('save.p')

result = net.predict(X_test)
print(result.argmax(axis=0))
print(Y_test.argmax(axis=0))
print(y_test)


# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# a = np.array([1,12,3,4,5,6,7,8,9])
# print(a.argmax(axis=0)+1)
