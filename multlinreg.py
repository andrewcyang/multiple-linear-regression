# Multiple Linear Regression
#
# andrewcyang
#
# Normalizes csv data and performs multiple linear
# regression with an arbitrary number of features.

import andrewcyang_linregres as lrr
import numpy                 as np
import statistics            as s

#Learning rate
a = 0.01
#Max number of iterations
max_it = 20000
#Precision
p = 0.1

#Load csv as mxn matrix
dset = np.loadtxt('filename.txt', delimiter = ',')
dim = np.shape(dset)
m, n = dim[0], dim[1]

#Initialize:
    #Feature matrix X
    #Vector y
    #Parameter th (theta)
X, y, th = [], [], []

for i in range(0, m):
    #Let first column of X be 1's
    x = [1]
    for j in range(0, n - 1):
        x.append(dset[i][j])
    X.append(x)
    y.append(dset[i][n - 1])
    
#Normalize features
v_avg, v_dev = [None], [None]
for i in range(1, n):
    col = []
    for j in range(0, m):
        col.append(X[j][i])
    v_avg.append(s.mean(col))
    v_dev.append(s.stdev(col))
for i in range(0, m):
    for j in range(1, n):
        X[i][j] = (X[i][j] - v_avg[j]) / v_dev[j]

#Set th to (1,1,...,1) by default
for i in range(0, n):
    th.append(1)

it = 0
p_curr = p + 1
while it < max_it and p_curr > p:
    old = lrr.loss(X, y, th)
    th = lrr.renew(th, lrr.mult(lrr.grad(X, y, th), a))
    p_curr = old - lrr.loss(X, y, th)
    it += 1

#Output
output = 'Regression given by:\nh = ' + str(round(th[0], 4))
for i in range(1, len(th)):
    output += ' + x' + str(i) + '(' + str(round(th[i], 4)) + ')'
print(output)
