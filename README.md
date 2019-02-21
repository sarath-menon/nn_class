 A simple neural network library written entirely in numpy.Intended mainly for educational purposes.
 
 # Features
 
 1)Easy to use API identical to pytorch
 2)Easy to add extra features like additional loss functions ,optimizers etc 

## A simple example 

 ```python
from nn_class_version_1 import nn
from helper_functions import *

train_x_orig, Y_train, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
X_train= train_x_flatten/255.


net = nn(X_train ,Y_train ,Plot_loss=True)
net.create_layer('Input',Relu)
net.create_layer(10 ,sigmoid)
net.create_model()

net.train()
```
