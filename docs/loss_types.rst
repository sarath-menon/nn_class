#########################
Types of Loss Functions
#########################


1) Mean squared error
=====================
 
 It is the most basic loss function .It just calculates the difference between the actual and prediced values and squares it.
 
Advantages
 --------------
 
 - Very easy to implement and understand
 - Squaring makes the loss is always positive and amplifies it 

Problems
 --------------
 
 - Difficult to find the lowest points as there are lots of locally low points ,which makes it very difficult backprop
 
 ::

    nn.loss_function(loss_function='Mean_squared_error')
    
Arguments  
 --------------
 
 - loss_type : Type of loss funtion
 - Plot loss == True : Plots how the loss varies as the network is trained  

2) Cross entropy loss
=====================
 
 Used for binary classificarion problems ,ie, prediciting the correct output label of input data which may belong to any of two input classes

Advantages
 --------------
 
 - Very easy to implement and understand
 - Squaring makes the loss is always positive and amplifies it 

Problems
 --------------
 
 - Difficult to find the lowest points as there are lots of locally low points ,which makes it very difficult backprop
 
 ::

    nn.loss_function(loss_function='Cross_entropy_loss')
    
Arguments  
 --------------
 
 - loss_type : Type of loss funtion
 - Plot loss == True : Plots how the loss varies as the network is trained  
