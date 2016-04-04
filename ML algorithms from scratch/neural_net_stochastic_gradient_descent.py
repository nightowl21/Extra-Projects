"""
Implemented a neural net with 3 layers for Regression.
Parameters as estimated with stochastic Gradient Descent.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    num_examples = X.shape[0]
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    y_hat = np.array([i[0] for i in z2])
    return np.sqrt(1./num_examples * np.sum(np.square(y-y_hat)))

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    return z2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(a):
    return sigmoid(a)*(1-sigmoid(a))

# PART1
def build_model(nn_hdim, XX, yy, num_passes=200, print_loss=True):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    loss =[]
    # Gradient descent. For each epoch...
    for j in xrange(1, num_passes+1):

        for ii in range(XX.shape[0]):
            # Forward propagation
            X = XX[ii]
            y = yy[ii]
            z1 = X.dot(W1) + b1
            a1 = sigmoid(z1)
            z2 = a1.dot(W2) + b2

            # Backpropagation
            y_hat = np.array([k[0] for k in z2])
            delta3 = -2*(y-y_hat)
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * sigmoid_prime(a1)
            dW1 = np.dot(X.T.reshape(2,1), delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 = dW2.reshape(nn_hdim,1)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss:
            print "Loss after iteration %d: %f" %(j, calculate_loss(model, XX, yy))
        loss.append(calculate_loss(model, XX, yy))

    return model, loss


a1 = np.array([3, 3])
a2 = np.array([3, -3])

Z_train = np.random.normal(size=100)
X_train = np.random.normal(size=(100,2))
Y_train = 1/(1+np.exp(np.sum(np.multiply(X_train, a1), axis=1))) + np.sum(np.square(np.multiply(X_train, a2)), axis=1) + 0.30*Z_train

Z_test = np.random.normal(size=1000)
X_test = np.random.normal(size=(1000,2))
Y_test = 1/(1+np.exp(np.sum(np.multiply(X_test, a1), axis=1))) + np.sum(np.square(np.multiply(X_test, a2)), axis=1) + 0.30*Z_test

nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 1 # output layer dimensionality
nn_hdim=10
reg_lambda = 0.1 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.001 # learning rate for gradient descent


model, loss = build_model(nn_hdim, X_train, Y_train, 400, print_loss=False)
preds = predict(model, X_test)
rmse = calculate_loss(model, X_test, Y_test)

