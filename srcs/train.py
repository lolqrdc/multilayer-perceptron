"""
Training module for the Multilayer Perceptron.
Handles network architecture, training (forward + backprop) and model saving.
"""

import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu':    (relu,    relu_derivative),
    'softmax': (softmax, None),
}

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        assert activation in ACTIVATIONS, f"Unknown activation: {activation}"

        limit = np.sqrt(2 / n_inputs)
        self.W = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        self.b = np.zeros((1, n_neurons))

        self.activation_name = activation
        self.activation, self.activation_derivative = ACTIVATIONS[activation]

        self.input  = None
        self.z      = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.output = self.activation(self.z)
        return self.output

def forward(network, X):
    output = X
    for layer in network:
        output = layer.forward(output)
    return output  

def categorical_crossentropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    labels      = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)

def backprop(network, y_pred, y_true, learning_rate):
    n = y_true.shape[0]
    delta = y_pred - y_true

    for i, layer in enumerate(reversed(network)):
        # Gradients pour cette couche
        dW = layer.input.T @ delta / n
        db = np.mean(delta, axis=0, keepdims=True)

        if i < len(network) - 1:
            delta_next = delta @ layer.W.T  # W original
            
            prev_layer = list(reversed(network))[i + 1]
            if prev_layer.activation_name != 'softmax':
                delta_next = delta_next * prev_layer.activation_derivative(prev_layer.z)
            
            delta = delta_next

        layer.W -= learning_rate * dW
        layer.b -= learning_rate * db

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)

    y_raw = df.iloc[:, 1].values
    X     = df.iloc[:, 2:].values.astype(float)

    y = np.zeros((len(y_raw), 2))
    y[y_raw == 'B', 0] = 1
    y[y_raw == 'M', 1] = 1

    return X, y

def normalize(X_train, X_valid):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_valid - mean) / std, mean, std

def train(network, X_train, y_train, X_valid, y_valid,
          learning_rate, epochs, batch_size):

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    n = X_train.shape[0]

    for epoch in range(1, epochs + 1):

        indices = np.random.permutation(n)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for start in range(0, n, batch_size):
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]

            y_pred = forward(network, X_batch)
            backprop(network, y_pred, y_batch, learning_rate)

        train_pred = forward(network, X_train)
        valid_pred = forward(network, X_valid)

        loss     = categorical_crossentropy(train_pred, y_train)
        val_loss = categorical_crossentropy(valid_pred, y_valid)
        acc      = accuracy(train_pred, y_train)
        val_acc  = accuracy(valid_pred, y_valid)

        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(acc)
        history['val_acc'].append(val_acc)

        print(f"epoch {epoch:02d}/{epochs} - "
              f"loss: {loss:.4f} - val_loss: {val_loss:.4f} - "
              f"acc: {acc:.4f} - val_acc: {val_acc:.4f}")

    return history

def plot_curves(history, save_path="outputs/learning_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['loss'],     label='training loss')
    ax1.plot(history['val_loss'], label='validation loss', linestyle='--')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    ax2.plot(history['acc'],     label='training acc')
    ax2.plot(history['val_acc'], label='validation acc', linestyle='--')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")

def save_history(history, filepath="outputs/history.json"):
    history_json = {
        key: [float(val) for val in values] 
        for key, values in history.items()
    }
    with open(filepath, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"Training history saved to {filepath}")

def load_history(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_model(network, mean, std, filepath="model.npy"):
    model = {
        'layers': [
            {
                'W':          layer.W,
                'b':          layer.b,
                'activation': layer.activation_name,
            }
            for layer in network
        ],
        'mean': mean,
        'std':  std,
    }
    np.save(filepath, model)
    print(f"> saving model '{filepath}' to disk...")

def main():
    parser = argparse.ArgumentParser(description="Train a multilayer perceptron")
    parser.add_argument("--train",         default="data/data_train.csv")
    parser.add_argument("--valid",         default="data/data_valid.csv")
    parser.add_argument("--layers",        type=int, nargs='+', default=[32, 32])
    parser.add_argument("--epochs",        type=int,   default=84)
    parser.add_argument("--model",         default="models/model.npy")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    
    if not args.model.startswith('models/'):
        args.model = os.path.join('models', os.path.basename(args.model))
    
    learning_rate = 0.0314
    batch_size = 8
    X_train, y_train = load_data(args.train)
    X_valid, y_valid = load_data(args.valid)
    X_train, X_valid, mean, std = normalize(X_train, X_valid)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    n_features = X_train.shape[1]
    n_classes  = 2                  

    network = []
    prev_size = n_features
    for size in args.layers:
        network.append(Layer(prev_size, size, activation='sigmoid'))
        prev_size = size
    network.append(Layer(prev_size, n_classes, activation='softmax'))

    history = train(network, X_train, y_train, X_valid, y_valid,
                    learning_rate=learning_rate,
                    epochs=args.epochs,
                    batch_size=batch_size)

    os.makedirs("models", exist_ok=True)
    save_model(network, mean, std, args.model)

    os.makedirs("outputs", exist_ok=True)
    plot_curves(history)
    
    history_path = args.model.replace('.npy', '_history.json')
    save_history(history, history_path)


if __name__ == "__main__":
    main()