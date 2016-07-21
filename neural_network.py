"""
Variables to be manually initialised:
    - lmbda
    - epsilon
    - hidden_layer_size
    - K
"""

import numpy as np
from scipy import optimize
from sklearn.preprocessing import scale
from sklearn import metrics

def featureNormalize(z):
    return scale(z)

def sigmoid(z):
    r = 1.0 / (1.0 + np.exp(-z))
    return r

def sigmoidGrad(z):
    r = sigmoid(z)
    r = r * (1.0 - r)
    return r

def randomizeTheta(l, epsilon):
    return ((np.random.random((l, 1)) * 2 * epsilon) - epsilon)

def KFoldDiv(X, y, m, n, K):
    sz = int(np.ceil(m / K))
    if n == 1:
        X_train = X[sz:, :]
        X_test = X[:sz, :]
        y_train = y[sz:]
        y_test = y[:sz]
    elif n == K:
        X_train = X[:((n-1)*sz), :]
        X_test = X[((n-1)*sz):, :]
        y_train = y[:((n-1)*sz)]
        y_test = y[((n-1)*sz):]
    else:
        X_train = np.vstack((X[:((n-1)*sz), :], X[(n*sz):, :]))
        X_test = X[((n-1)*sz):(n*sz), :]
        y_train = np.vstack((y[:((n-1)*sz)], y[(n*sz):]))
        y_test = y[((n-1)*sz):(n*sz)]
    return (X_train, y_train, X_test, y_test)

def nnCostFunc(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))
    m = X.shape[0]
    y = (y == np.array([(i+1) for i in range(num_labels)])).astype(int)

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    h = sigmoid(np.dot(a2, Theta2.T))

    cost = ((lmbda/2)*(np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)) - np.sum((y * np.log(h)) + ((1-y) * np.log(1-h)))) / m
    return cost

def nnGrad(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))
    m = X.shape[0]
    y = (y == np.array([(i+1) for i in range(num_labels)])).astype(int)

    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack((np.ones((m, 1)), sigmoid(z2)))
    h = sigmoid(np.dot(a2, Theta2.T))

    delta_3 = h - y
    delta_2 = np.dot(delta_3, Theta2[:, 1:]) * sigmoidGrad(z2)
    Theta2_grad = (np.dot(delta_3.T, a2) + (lmbda * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:])))) / m
    Theta1_grad = (np.dot(delta_2.T, a1) + (lmbda * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:])))) / m

    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad


K = 10
lmbda = 0.03
epsilon = 0.12

input_layer_size = 13
hidden_layer_size = 20
num_labels = 2

X = np.genfromtxt('heart.csv', delimiter=',')
m, n = X.shape
n -= 1

y = X[:, n].astype(int).reshape((m, 1))
X = featureNormalize(X[:, :n])
foldAcc = np.ndarray((K, 1))

FP = 0
FN = 0
TN = 0
TP = 0

for i in range(K):
    X_train, y_train, X_test, y_test = KFoldDiv(X, y, m, i+1, K)
    
    initTheta = randomizeTheta((hidden_layer_size * (input_layer_size+1)) + (num_labels * (hidden_layer_size+1)), epsilon)
    Theta = optimize.fmin_bfgs(nnCostFunc, initTheta, fprime=nnGrad, args=(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lmbda), maxiter=3000)
    Theta1, Theta2 = np.split(Theta, [hidden_layer_size * (input_layer_size+1)])
    Theta1 = np.reshape(Theta1, (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(Theta2, (num_labels, hidden_layer_size+1))

    h1 = sigmoid(np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), Theta1.T))
    h2 = sigmoid(np.dot(np.hstack((np.ones((h1.shape[0], 1)), h1)), Theta2.T))
    predicted = h2.argmax(1) + 1
    predicted = predicted.reshape((predicted.shape[0], 1))
    foldAcc[i] = np.mean((predicted == y_test).astype(float)) * 100

    cm = (metrics.confusion_matrix(y_test, predicted))/len(y_test)

    FP += cm[0][0]
    FN += cm[1][0]
    TN += cm[0][1]
    TP += cm[1][1]

    print('Test Set Accuracy for %dth fold: %f\n' % (i+1, foldAcc[i]))

meanAcc = np.mean(foldAcc)
print('\nAverage Accuracy: ', meanAcc)
print("")

print(FP)
print(FN)
print(TN)
print(TP)