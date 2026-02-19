"""
Prediction module for the MLP.
Loads a trained model and evaluates it on a test set using binary cross entropy.
"""

import numpy as np
import argparse
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'relu':    lambda z: np.maximum(0, z),
    'softmax': softmax,
}

class Layer:
    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        z = x @ self.W + self.b
        return self.activation(z)

def load_model(filepath):
    model = np.load(filepath, allow_pickle=True).item()
    
    network = [
        Layer(layer['W'], layer['b'], layer['activation'])
        for layer in model['layers']
    ]
    
    mean = model['mean']
    std  = model['std']
    
    return network, mean, std

def forward(network, X):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output

def binary_crossentropy(y_pred, y_true):
    p = y_pred[:, 1]
    y = y_true[:, 1]  # 1 for M, 0 for B
    
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    labels      = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)


def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    
    y_raw = df.iloc[:, 1].values
    X     = df.iloc[:, 2:].values.astype(float)
    
    # one-hot encode: B → [1, 0], M → [0, 1]
    y = np.zeros((len(y_raw), 2))
    y[y_raw == 'B', 0] = 1
    y[y_raw == 'M', 1] = 1
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Predict with trained MLP")
    parser.add_argument("--model",   default="models/model.npy")
    parser.add_argument("--dataset", default="data/data_valid.csv")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    network, mean, std = load_model(args.model)
    print(f"Model loaded: {len(network)} layers")

    X_test, y_test = load_data(args.dataset)
    print(f"Test data shape: {X_test.shape}")
    
    X_test = (X_test - mean) / std

    y_pred = forward(network, X_test)

    loss = binary_crossentropy(y_pred, y_test)
    acc  = accuracy(y_pred, y_test)

    print(f"\nBinary Cross-Entropy Loss: {loss:.6f}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()