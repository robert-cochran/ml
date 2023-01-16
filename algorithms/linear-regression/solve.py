def solve(X, Y):
    denominator = X.dot(X) - X.mean() * X.sum()
    a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
    b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
    
    return a, b


def solve_direct(X,Y):
    X = X.reshape((len(X), 1))
    # linear least squares
    b = inv(X.T.dot(X)).dot(X.T).dot(y)
    print(b)
    # predict using coefficients
    yhat = X.dot(b)



def solve_QRDecomp():
    print("S")



def solve_SVD():
    print("Singular-Value Decomposition")


