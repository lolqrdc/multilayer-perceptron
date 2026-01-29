#GOAL: Train MLP on the dataset using backpropagation and gradient descent.
# 2 hidden layers minimum, softmax function.

import csv
import os
import numpy as np

def readData(data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mlp_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(mlp_dir, "data")
    filepath = os.path.join(data_dir, data)

    features = []
    diagnosis = []
    with open(filepath, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            diagnosis.append(row[1])
            features.append([float(x) for x in row[2:]])
    return np.array(features), np.array(diagnosis)

def binary_labels(y_str):
    return np.array([1 if label == 'M' else 0 for label in y_str])

def normalize(X_train, X_valid):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_trainNorm = (X_train - mean) / std
    X_validNorm = (X_valid - mean) / std
    return (X_trainNorm, X_validNorm, mean, std)

class MLP:
    def __init__(self, layer_sizes=[30, 24, 24, 2]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.bias = []
        self.build()

    def build(self):
        for i in range(1, len(self.layer_sizes)):
            fan_in = self.layer_sizes[i-1]
            fan_out = self.layer_sizes[i]

            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in) # Poids (W): l'importance attribue a chaque entree
            b = np.zeros((1, fan_out)) # Biais (b): aide le neurone a s'activer ou a rester inactif independamment des entrees
            self.weights.append(W)
            self.bias.append(b)
    
    def relu(self, z): # funct d'activation: declenche le neurone si Z > 0 (ex: detect une texture granuleuse = actif)
        return (np.maximum(0, z))

    def softmax(self, z): # transformer les scores bruts en %
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return (exp_z / np.sum(exp_z, axis=1, keepdims=True))
         
    def feedforward(self, X): # X = m patients x 30 mesures chacun
        caches = {}
        A = X
        
        for l in range(1, len(self.layer_sizes)): #
            Z = A @ self.weights[l-1] + self.bias[l-1]

            if l == len(self.layer_sizes) - 1: # Derniere couche
                A = self.softmax(Z) # Transformation du scores brut en proba
            else:
                A = self.relu(Z)

            caches[f'Z{l}'] = Z # Save du score brut 
            caches[f'A{l}'] = A # Save de la proba

        return (A, caches)

def main():

# Phase 1 load data
    X_trainRow, y_trainStr = readData("data_train.csv")
    X_ValidRow, y_validStr = readData("data_valid.csv")

# Phase 2 preprocess 
    y_train = binary_labels(y_trainStr)
    y_valid = binary_labels(y_validStr)
    X_train, X_valid, mean, std = normalize(X_trainRow, X_ValidRow)

# TEST BY PRINT   
    print(f"RAW shapes: train {X_trainRow.shape}, valid {X_ValidRow.shape}")
    print(f"NORM shapes: train {X_train.shape}, valid {X_valid.shape}")
    print(f"Labels train: {np.sum(y_train)} M / {len(y_train)} total")
    print(f"radius_mean après norm: mean={np.mean(X_train[:,0]):.3f}")

    mlp = MLP([30, 24, 24, 2])
    Y_pred, caches = mlp.feedforward(X_train[:3])
    print(f"X: {X_train[:3].shape}")
    print(f"Y_pred:{Y_pred.shape}")
    print(f"Probas:\n{Y_pred.round(3)}")
    print(f"Somme=1: {np.sum(Y_pred, axis=1).round(3)}")
    print(f"Cache: {list(caches.keys())}")

if __name__ == "__main__":
    main()
