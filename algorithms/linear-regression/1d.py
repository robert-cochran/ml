# This script uses code from lazyprogrammer/machine-learning-examples and https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/

#
import numpy as np
import matplotlib.pyplot as plt
import solve

def load_data(filename):
    X = []
    Y = []
    for line in open(filename):
        x, y = line.split(',')
        X.append(float(x))
        Y.append(float(y))
    
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


if  __name__ == "__main__":
    #load the data
    X, Y = load_data('./data/data_1d.csv')


    #plot the model to ensure its been loaded correctly
    plt.scatter(X, Y)
    plt.show() 

    a, b = solve.solve(X,Y)
    print(a, b)

    #define/create the abstract model

    #solve the model

    #use model to generate yhat across all X

    #find difference between yhat and X

    #display the model results as a function

    #display r-squared

    #display the resutls as a plot



