# Dataset taken from Andrew NG's Deep learning course on coursera
import numpy as np

def softmax(actions):
    actions_soft = np.divide(np.exp(actions),np.sum(np.exp(actions)))
    # for i in range(actions):
    return actions_soft

a = np.array([5,2,-1,3])
b = softmax(a)
print(b)
