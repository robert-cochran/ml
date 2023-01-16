import numpy as np
import logging

logging.basicConfig(filename='log.txt', filemode='w', level=logging.INFO, format='%(message)s')

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1 #synapse
syn1 = 2*np.random.random((4,1)) - 1

logging.info("syn0: " + str(syn0))
logging.info("syn1: " + str(syn1))
logging.info("np.dot(X,syn0): " + str(np.dot(X,syn0)))

sigmoid = lambda x: 1/(1+np.exp(-(x)))
backprp = lambda x: x * (1-x)

for j in xrange(60000):
    l1 = sigmoid(np.dot(X,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    if (j % 10000) == 0:
	logging.info("error: " + str(np.mean(np.abs((y-l2))))) 
    #l2_delta = (y - l2)*(l2*(1-l2)) 
    l2_delta = (y-l2) * backprp(l2)
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

logging.info("l1: " + str(l1))
logging.info("l2: " + str(l2))

