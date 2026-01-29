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

if __name__ == "__main__":
    main()
