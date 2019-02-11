# # Dataset taken from Andrew NG's Deep learning course on coursera
from nn_class import nn
from helper_functions import *
import matplotlib.pyplot as plt
import random
import keras
from keras.datasets import mnist
random.seed(1)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(28*28 ,60000)

Y_train = keras.utils.to_categorical(y_train, 10).T
print('Xshape',X_train.shape)
print('yshape',Y_train.shape)
del mnist

net = nn(X_train.shape[0] ,Y_train.shape[0] ,Plot_loss=True)
net.create_layer('Input',Relu)
net.create_layer(256 ,softmax)
net.create_model()
net.train(X_train ,Y_train,learning_rate=0.01,epochs=20)

X_test = x_test.reshape(28*28 ,10000)
Y_test = keras.utils.to_categorical(y_test, 10).T

print('Xshape',X_test.shape)
print('yshape',Y_test.shape)
result = net.predict(X_test)
print(result.argmax(axis=0)+1)
print(Y_test.argmax(axis=0)+1)

# result = net.predict(X_train)
# print(result.shape)
# a = np.array([[1,2,3,5],[4,5,6,7],[7,8,9,9]])
# a = np.array([1,2,3])
# print(np.sum(a,axis=0))
# print(softmax(a))
