import numpy as np

def neural_net():
    def __init__(x, r, nodes_per_layer):
        self.input_layer = []
        self.input_layer.append(x)
        self.r = r
        self.nodes_per_layer = nodes_per_layer
    
    def train(x, r, nodes_per_layer):
        w = init_weights(w, nodes_per_layer)

        for i in range(len(nodes_per_layer)):
            #maps input to value between 0 and 1
            a = activation(w * input_layer[i]) 

            #determines if this node was activated
            h = threshold(a)

            #stores the result to be called later 
            input_layer.append(h)

        y = input_layer[-1]

        cost = cost(r, y)

        w_backprop = backpropogation(w, cost)

        #returns a value,
        #does this value match our R for the corresponding x
        #this determines our 
        

        return w_backprop
    

    def init_weights(w, nodes_per_layer):
        w = np.zeroes(len(x))
        return w
    

    def activation():
        return 0

    def threshold():
        return 0

    def cost():
        return 0

    def backpropogation():
        return 0