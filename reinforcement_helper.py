import numpy as np

def random_sampling(action_space):
    action = np.random.randint(action_space)
    return action


def gradient_ascent(self ,optimizer ,beta):
    for l in range(len(self.hidden_layers)+1):
        if (optimizer == 'Momentum'):
            if ("v_dW" + str(l+1)) in self.v_grad != None :
                self.v_grad["v_dW" + str(l+1)] = (beta * self.v_grad["v_dW" + str(l+1)]) +  0.9*self.grads["dW" + str(l+1)]
                self.v_grad["v_db" + str(l+1)] = (beta * self.v_grad["v_db" + str(l+1)]) +  0.9*self.grads["db" + str(l+1)]

                self.parameters["W" + str(l+1)] += self.learning_rate * self.v_grad["v_dW" + str(l+1)]
                self.parameters["b" + str(l+1)] += self.learning_rate * self.v_grad["v_db" + str(l+1)]

            else:self.v_grad["v_dW" + str(l+1)] ,self.v_grad["v_db" + str(l+1)] = 0 ,0

        if (optimizer == 'Gradient_descent'):
            self.parameters["W" + str(l+1)] += self.learning_rate * self.grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] += self.learning_rate * self.grads["db" + str(l+1)]
