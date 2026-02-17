"""
Multilayer Perceptron (MLP) Core Module

Core neural network implementation for breast cancer classification 
(30 features → 2 classes: M/B). Implements feedforward propagation, 
backpropagation, and gradient descent from scratch with sigmoid 
hidden layers and softmax output.
"""

import numpy as np

def init_network(layer_sizes):
    params = {}
    L = len(layer_sizes)

    for l in range(1, L):
        n_in = layer_sizes[l-1]
        n_out = layer_sizes[l]

        params[f'W{l}'] = np.random.randn(n_out, n_in) * 0.01
        params[f'b{l}'] = np.zeros((n_out, 1))

    return (params)

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def sigmoid_derivate(z):
    s = sigmoid(z)
    return (s * (1 - s))

def forward(X, params):
    cache = {}
    L = len(params) // 2 # nb total de couches (hidden, output)
    cache['a0'] = X

    for l in range(1, L + 1): # parcourir l'ensemble des layers
        cache[f'z{l}'] = params[f'W{l}'] @ cache[f'a{l-1}'] + params[f'b{l}']

        if l < L: # si hidden layer appliquer funct sigmoid
            cache[f'a{l}'] = sigmoid(cache[f'z{l}'])
        else:
            pass

    y_hat = cache['z3']
    return (y_hat, cache)

def softmax(z): # introduction de la non linearite 
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    return (exp_z / np.sum(exp_z, axis=0, keepdims=True))

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))

def backpropagation(X, y_true, params, cache):
    L = len(params) // 2  
    batch_size = X.shape[1]
    y_pred = softmax(cache[f'z{L}'])
    gradients = {}
    
    dz = y_pred - y_true
    gradients[f'dz{L}'] = dz
    gradients[f'dW{L}'] = (1/batch_size) * dz @ cache[f'a{L-1}'].T
    gradients[f'db{L}'] = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)
    
    for l in reversed(range(1, L)):
        da = params[f'W{l+1}'].T @ dz  
        
        dz = da * sigmoid_derivate(cache[f'z{l}'])
        
        gradients[f'dz{l}'] = dz
        gradients[f'dW{l}'] = (1/batch_size) * dz @ cache[f'a{l-1}'].T
        gradients[f'db{l}'] = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)
    
    return (gradients)

def gradient_descent(params, gradient, learning_rate):
    gradient_descent = params.copy()
    
    for l in range(1, len(params) // 2 + 1):
        gradient_descent[f'W{l}'] -= learning_rate * gradient[f'dW{l}']
        gradient_descent[f'b{l}'] -= learning_rate * gradient[f'db{l}']

    return (gradient_descent)

if __name__ == "__main__":
    arch = [30, 24, 24, 2]
    params = init_network(arch)
    
    X_test = np.random.randn(30, 2)
    y_true = np.array([[1, 0], [0, 1]])  # M, B
    
    print("=== TRAINING 1 EPOCH ===")
    print("Before training:")
    
    # Forward before
    y_hat, cache = forward(X_test, params)
    probas = softmax(y_hat)
    loss1 = cross_entropy(y_true, probas)
    print(f"Loss initiale: {loss1:.4f}")
    
    # Backpropagation + Update
    grads = backpropagation(X_test, y_true, params, cache)
    params = gradient_descent(params, grads, learning_rate=0.1)
    
    # Forward after
    y_hat2, _ = forward(X_test, params)
    probas2 = softmax(y_hat2)
    loss2 = cross_entropy(y_true, probas2)
    print(f"Loss après 1 époque: {loss2:.4f}")
    print(f"Amélioration: {loss1 - loss2:.4f}")

