#########################
Loss Functions
#########################


In the previous section ,we talked about forward prop which takes in the data as input and outputs bullshit in early stages of trainig.Loss functions is the first step of actual 'Learning' in a neural network. **It measures the difference between the actual ouput given in the dataset and the wrong output predicted by the neural network .Based on how big the loss is ,we have to modify the parameters of the network so that the loss is minimised.**

::

    nn.loss_function(Plot_loss)
    
Arguments  
 --------------
 
 - loss_type : Type of loss funtion
 - Plot loss == True : Plots how the loss varies as the network is trained  
 
 
 .. toctree::
   :maxdepth: 1
   :titlesonly:
   
   loss_types
