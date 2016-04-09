import numpy as np
from scipy.optimize import fmin_cg

def sigmoid(z):
    return 1./(1+np.exp(-z))

def prediction(X, theta):
    return sigmoid(np.dot(X, theta))

def costfunction(theta, *args):
    X, y, lmbda = args
    m = y.size
    n = theta.size
    pred = prediction(X, theta)
    cost = (-1./m)*np.sum(y*np.log(pred)+(1-y)*np.log(1-pred))
    return cost + (lmbda/(2.*m))*np.sum(np.square(theta[1:n]))

def grad(theta, *args):
    X, y, lmbda = args
    m = y.size
    n = theta.size
    pred = prediction(X, theta)
    g = np.sum((pred.reshape((pred.size,1))-y.reshape((y.size, 1)))*X, axis=0)
    g[1:] = g[1:] + (lmbda/m)*theta[1:]
    return g

def onevall(X, y, lmbda):
    tempX = np.ones((X.shape[0], X.shape[1] + 1))
    tempX[:, 1:] = X
    X = tempX
    y = y.reshape(y.size)
    inittheta = np.zeros(X.shape[1])
    lmbda = 0.1
    ys = set(y)
    alltheta = np.ones((len(ys), inittheta.size))*0.5
    for i in ys:
        tempY = np.zeros(y.shape)
        tempY[np.where(y==i)] = 1
        args = (X,tempY,lmbda)
        theta = fmin_cg(costfunction, inittheta, fprime=grad, args=args, maxiter=50)
        if i == 10:
            i = 0
        alltheta[i] = theta
    return alltheta
