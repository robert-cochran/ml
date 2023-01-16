import numpy as np

X = np.array( [	[0,0,1],
		[0,1,1],
            	[1,0,1],
           	[1,1,1] ] )
                
y = np.array( [ [0],
		[1],
		[1],
		[0] ] )

syn0 = np.random.random([3,4])
syn1 = np.random.random([4,1])

#def nn(nonlinear, X, y):
#	return lambda x: f1(f2(x))

for x in range(2):
	l1 = 1/(1+(np.exp(-(np.dot(X,syn0)))))
	l2 = 1/(1+(np.exp(-(np.dot(l1,syn1)))))

