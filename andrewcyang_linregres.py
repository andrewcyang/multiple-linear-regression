# Calculations for Multiple Linear Regression
#
# andrewcyang
#
# Normalizes csv data and performs multiple linear
# regression with an arbitrary number of features.
#
# Loss function:
#     J(theta) = (1/2m)sum[i=1 to m](h(x^(i))-y^(i))^2
#     m : number of training examples
#     h(x^(i)) : hypothesis on the ith training example

import numpy as np
import statistics as s

#Returns difference of h(x) and y
def err(x, y, th):
    return (np.dot(x, th) - y)

#Returns J(current theta)
def loss(X, y, th):
    e, m = 0, len(X)
    for i in range(0, m):
        e += (err(X[i], y[i], th) ** 2)
    return (e / (2 * m))

#Returns partial derivative of J wrt theta_k
def deriv(X, y, th, k):
    e, m = 0, len(X)
    for i in range(0, m):
        e += (err(X[i], y[i], th) * X[i][k])
    return (e / len(X))

#Returns partial derivatives of J as list
def grad(X, y, th):
    g = []
    for k in range(0, len(th)):
        g.append(deriv(X, y, th, k))
    return g

#Multiplies a list g by a scalar a
def mult(g, a):
    for i in range(0, len(g)):
        g[i] *= a
    return g

#Returns updated theta (gradient descent)
def renew(th, g):
    for i in range(0, len(th)):
        th[i] -= g[i]
    return th
