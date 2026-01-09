import numpy as np

# sigmoid, sigmoid_derivative, softmax, init_weights, classe MLP

def sigmoid(z):
    z_safe = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_safe))

def sigmoid_derive(a):
    return a * (1 - a)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def init_weights(n_in, n_out):
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(-limit, limit, (n_in, n_out))

class MLP:
    
    def __init__(self, layer_sizes):
        self.weights = []
        self.bias = []

        for i in range(len(layer_sizes) - 1):
            W = init_weights(layer_sizes[i], layer_sizes[i+1])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.bias.append(b)

    def forward(self, X): # Passer d'une couche a une autre
        self.A = [X] 
        
        for i in range(len(self.weights)):
            # z = X @ W + b
            z = self.A[-1] @ self.weights[i] + self.bias[i]
            
            # Activation
            if i == len(self.weights) - 1:  # last couche
                a = softmax(z)
            else:  # Couches hidden
                a = sigmoid(z)
            
            self.A.append(a)
        
        return self.A[-1]
    
    def backward(self, y_onehot, lr):
        m = y_onehot.shape[0]
        
        # Erreur de sortie
        delta = self.A[-1] - y_onehot
        
        # Parcourir les couches dans l'autre sens
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients
            dW = self.A[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            # Propager l'erreur vers la couche d'avant
            if i > 0:
                delta = (delta @ self.weights[i].T) * sigmoid_derive(self.A[i])
            
            # Mise à jour des poids (descente de gradient)
            self.weights[i] -= lr * dW
            self.bias[i] -= lr * db
    
    def predict(self, X):
